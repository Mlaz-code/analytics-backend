#!/bin/bash
# =============================================================================
# Pikkit ML API - Deployment Script
# Supports blue-green deployment with health checks and rollback
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"
ENV_FILE="$PROJECT_DIR/.env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Load environment
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# Default values
: "${APP_VERSION:=1.0.0}"
: "${HEALTH_CHECK_TIMEOUT:=60}"
: "${HEALTH_CHECK_INTERVAL:=5}"
: "${MODEL_PATH:=/root/pikkit/ml/models}"

export APP_VERSION
export MODEL_PATH
export BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export GIT_COMMIT="$(git -C "$PROJECT_DIR" rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# Get current active deployment
get_active_deployment() {
    local blue_healthy green_healthy

    blue_healthy=$(docker inspect --format='{{.State.Health.Status}}' pikkit-ml-api-blue 2>/dev/null || echo "none")
    green_healthy=$(docker inspect --format='{{.State.Health.Status}}' pikkit-ml-api-green 2>/dev/null || echo "none")

    if [[ "$blue_healthy" == "healthy" ]]; then
        echo "blue"
    elif [[ "$green_healthy" == "healthy" ]]; then
        echo "green"
    else
        echo "none"
    fi
}

# Get inactive deployment
get_inactive_deployment() {
    local active
    active=$(get_active_deployment)

    if [[ "$active" == "blue" ]]; then
        echo "green"
    else
        echo "blue"
    fi
}

# Health check function
health_check() {
    local service=$1
    local port=$2
    local max_attempts=$((HEALTH_CHECK_TIMEOUT / HEALTH_CHECK_INTERVAL))
    local attempt=0

    log_info "Waiting for $service to become healthy..."

    while [[ $attempt -lt $max_attempts ]]; do
        if curl -sf "http://localhost:$port/ready" > /dev/null 2>&1; then
            log_success "$service is healthy"
            return 0
        fi

        attempt=$((attempt + 1))
        log_info "Health check attempt $attempt/$max_attempts..."
        sleep "$HEALTH_CHECK_INTERVAL"
    done

    log_error "$service failed health checks after ${HEALTH_CHECK_TIMEOUT}s"
    return 1
}

# Build new image
build_image() {
    log_info "Building Docker image version $APP_VERSION..."

    docker-compose -f "$COMPOSE_FILE" build \
        --build-arg APP_VERSION="$APP_VERSION" \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg GIT_COMMIT="$GIT_COMMIT" \
        pikkit-ml-api-blue

    log_success "Image built successfully"
}

# Blue-green deployment
deploy_blue_green() {
    local active inactive port

    active=$(get_active_deployment)
    inactive=$(get_inactive_deployment)

    log_info "Current active: $active"
    log_info "Deploying to: $inactive"

    # Determine port for inactive deployment
    if [[ "$inactive" == "blue" ]]; then
        port=8001
    else
        port=8002
    fi

    # Build new image
    build_image

    # Deploy to inactive slot
    log_info "Starting $inactive deployment..."
    docker-compose -f "$COMPOSE_FILE" --profile blue-green up -d "pikkit-ml-api-$inactive"

    # Wait for health
    if ! health_check "pikkit-ml-api-$inactive" "$port"; then
        log_error "Deployment failed! Rolling back..."
        docker-compose -f "$COMPOSE_FILE" stop "pikkit-ml-api-$inactive"
        exit 1
    fi

    # Switch traffic (update nginx upstream or use docker labels)
    log_info "Switching traffic to $inactive..."

    # Reload nginx to pick up new healthy container
    docker exec pikkit-nginx-lb nginx -s reload 2>/dev/null || true

    # Gracefully stop old deployment
    if [[ "$active" != "none" ]]; then
        log_info "Stopping $active deployment..."
        sleep 10  # Allow in-flight requests to complete
        docker-compose -f "$COMPOSE_FILE" stop "pikkit-ml-api-$active"
    fi

    log_success "Blue-green deployment complete! Active: $inactive"
}

# Canary deployment
deploy_canary() {
    local canary_version=${1:-"$APP_VERSION-canary"}

    log_info "Deploying canary version: $canary_version"

    export CANARY_VERSION="$canary_version"
    export CANARY_MODEL_PATH="${CANARY_MODEL_PATH:-$MODEL_PATH}"

    # Build canary image
    docker-compose -f "$COMPOSE_FILE" build \
        --build-arg APP_VERSION="$canary_version" \
        pikkit-ml-api-canary

    # Start canary
    docker-compose -f "$COMPOSE_FILE" --profile canary up -d pikkit-ml-api-canary

    # Wait for health
    if ! health_check "pikkit-ml-api-canary" 8003; then
        log_error "Canary deployment failed!"
        docker-compose -f "$COMPOSE_FILE" --profile canary down
        exit 1
    fi

    log_success "Canary deployed successfully on port 8003"
    log_info "Monitor metrics at http://localhost:8003/metrics"
}

# Promote canary to production
promote_canary() {
    log_info "Promoting canary to production..."

    # Get canary version
    local canary_version
    canary_version=$(docker inspect --format='{{.Config.Env}}' pikkit-ml-api-canary 2>/dev/null | grep -oP 'APP_VERSION=\K[^,\]]+' || echo "unknown")

    export APP_VERSION="$canary_version"

    # Deploy via blue-green
    deploy_blue_green

    # Stop canary
    docker-compose -f "$COMPOSE_FILE" --profile canary down

    log_success "Canary promoted to production"
}

# Rollback to previous version
rollback() {
    local current inactive

    current=$(get_active_deployment)
    inactive=$(get_inactive_deployment)

    log_warning "Rolling back from $current to $inactive..."

    # Start inactive (previous) deployment
    docker-compose -f "$COMPOSE_FILE" --profile blue-green up -d "pikkit-ml-api-$inactive"

    # Determine port
    local port
    if [[ "$inactive" == "blue" ]]; then
        port=8001
    else
        port=8002
    fi

    # Wait for health
    if ! health_check "pikkit-ml-api-$inactive" "$port"; then
        log_error "Rollback failed! Manual intervention required."
        exit 1
    fi

    # Switch traffic
    docker exec pikkit-nginx-lb nginx -s reload 2>/dev/null || true

    # Stop current
    docker-compose -f "$COMPOSE_FILE" stop "pikkit-ml-api-$current"

    log_success "Rollback complete! Active: $inactive"
}

# Status check
status() {
    echo ""
    log_info "=== Pikkit ML API Deployment Status ==="
    echo ""

    local active
    active=$(get_active_deployment)
    echo "Active Deployment: $active"
    echo ""

    echo "Container Status:"
    docker ps --filter "name=pikkit-ml" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""

    echo "Health Checks:"
    for service in blue green canary; do
        local container="pikkit-ml-api-$service"
        local status
        status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "not running")
        echo "  $service: $status"
    done
    echo ""
}

# Print usage
usage() {
    cat << EOF
Pikkit ML API Deployment Script

Usage: $0 <command> [options]

Commands:
    deploy          Deploy new version using blue-green strategy
    canary          Deploy canary version (10% traffic)
    promote         Promote canary to production
    rollback        Rollback to previous version
    status          Show deployment status
    build           Build Docker image only
    start           Start all services
    stop            Stop all services
    logs            Show container logs

Options:
    --version, -v   Specify version to deploy (default: $APP_VERSION)
    --help, -h      Show this help message

Examples:
    $0 deploy                     # Deploy current version
    $0 deploy --version 1.2.0     # Deploy specific version
    $0 canary --version 1.2.0-rc1 # Deploy canary
    $0 promote                    # Promote canary
    $0 rollback                   # Rollback to previous

EOF
}

# Main entry point
main() {
    local command=${1:-""}
    shift || true

    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --version|-v)
                APP_VERSION="$2"
                shift 2
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                shift
                ;;
        esac
    done

    case $command in
        deploy)
            deploy_blue_green
            ;;
        canary)
            deploy_canary "$APP_VERSION"
            ;;
        promote)
            promote_canary
            ;;
        rollback)
            rollback
            ;;
        status)
            status
            ;;
        build)
            build_image
            ;;
        start)
            docker-compose -f "$COMPOSE_FILE" up -d
            ;;
        stop)
            docker-compose -f "$COMPOSE_FILE" down
            ;;
        logs)
            docker-compose -f "$COMPOSE_FILE" logs -f
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

main "$@"
