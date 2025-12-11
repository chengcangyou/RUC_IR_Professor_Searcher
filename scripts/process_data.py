#!/usr/bin/env python3
"""
Simple data processor for professor information
Merges CSV files and generates JSON for the search engine
"""

import pandas as pd
import json
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Ensure processed directory exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_countries():
    """Load country information"""
    countries_file = RAW_DIR / "countries.csv"
    countries = {}
    
    df = pd.read_csv(countries_file)
    for _, row in df.iterrows():
        countries[row['name']] = {
            'alpha_2': row['alpha_2'],
            'region': row['region'],
            'sub_region': row['sub_region']
        }
    
    return countries


def load_institutions():
    """Load institution information"""
    institutions_file = RAW_DIR / "institutions.csv"
    institutions = {}
    
    df = pd.read_csv(institutions_file)
    for _, row in df.iterrows():
        institutions[row['institution']] = {
            'country': row['region']
        }
    
    return institutions


def process_csrankings():
    """Process CSRankings data"""
    print("Loading CSRankings data...")
    
    # Load main file
    csrankings_file = RAW_DIR / "csrankings.csv"
    df = pd.read_csv(csrankings_file)
    
    print(f"Loaded {len(df)} records")
    
    # Load countries and institutions
    countries = load_countries()
    institutions = load_institutions()
    
    # Process each professor
    professors = []
    seen_names = set()
    
    for idx, row in df.iterrows():
        name = row['name']
        
        # Skip duplicates
        if name in seen_names:
            continue
        seen_names.add(name)
        
        # Get institution info
        institution = row['affiliation']
        country_code = institutions.get(institution, {}).get('country', 'unknown')
        
        # Get country info
        country_info = None
        for country_name, info in countries.items():
            if info['alpha_2'].lower() == country_code.lower():
                country_info = info
                break
        
        professor = {
            'id': idx + 1,
            'name': name,
            'institution': institution,
            'region': country_info['region'].lower().replace(' ', '_') if country_info else 'unknown',
            'country': country_code.lower(),
            'countryName': country_code.upper(),
            'homepage': row.get('homepage', ''),
            'scholarId': row.get('scholarid', ''),
            'researchAreas': [],
            'publications': 0,
            'recentYear': 0
        }
        
        professors.append(professor)
    
    print(f"Processed {len(professors)} unique professors")
    
    # Save to JSON
    output_file = PROCESSED_DIR / "professors.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(professors, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    """Main function"""
    print("="*80)
    print("Simple Data Processor")
    print("="*80)
    print()
    
    process_csrankings()
    
    print()
    print("="*80)
    print("Processing complete!")
    print("="*80)


if __name__ == "__main__":
    main()

