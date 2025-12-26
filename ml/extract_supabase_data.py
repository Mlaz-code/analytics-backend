#!/usr/bin/env python3
"""Extract bet data from Supabase MCP result file"""

import json
import pandas as pd
import re

# Read the MCP result file
with open('/root/.claude/projects/-root-pikkit-ml/3cd72e1a-7979-4ed8-823b-75d28c347664/tool-results/mcp-supabase-execute_sql-1766587470775.txt', 'r') as f:
    data = json.load(f)

# Extract the text field
text_content = data[0]['text']

# Find the JSON array (it's between untrusted-data tags)
match = re.search(r'<untrusted-data-[^>]+>\n(.+)\n</untrusted-data', text_content, re.DOTALL)

if match:
    json_str = match.group(1)
    bets = json.loads(json_str)

    print(f"Extracted {len(bets)} bets")

    # Convert to DataFrame
    df = pd.DataFrame(bets)

    # Save to CSV
    output_path = '/root/pikkit/ml/data/all_bets_training.csv'
    df.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")
    print(f"\nSummary:")
    print(f"  Total bets: {len(df)}")
    print(f"  Sports: {df['sport'].nunique()} unique")
    print(f"  Leagues: {df['league'].nunique()} unique")
    print(f"  Markets: {df['market'].nunique()} unique")
    print(f"\nTop 10 sports:")
    print(df['sport'].value_counts().head(10))
    print(f"\nTop 10 markets:")
    print(df['market'].value_counts().head(10))
else:
    print("Could not find JSON data in file")
