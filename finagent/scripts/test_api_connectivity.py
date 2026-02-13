#!/usr/bin/env python3
"""
Test script to diagnose BSE/NSE API connectivity issues.
Run: python -m finagent.scripts.test_api_connectivity
"""

import requests
import time
import sys

# BSE Headers
BSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Referer': 'https://www.bseindia.com/',
    'Origin': 'https://www.bseindia.com',
}

# NSE Headers (no brotli - 'br' requires extra library)
NSE_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
    'Referer': 'https://www.nseindia.com/',
    'Origin': 'https://www.nseindia.com',
    'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Fetch-Dest': 'empty',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Site': 'same-origin',
}


def test_bse_api():
    """Test BSE API connectivity."""
    print("\n" + "="*60)
    print("Testing BSE API")
    print("="*60)

    from datetime import datetime, timedelta
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
    to_date = datetime.now().strftime('%Y%m%d')

    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnGetData/w"
    params = {
        'strCat': '-1',
        'strPrevDate': from_date,
        'strScrip': '',
        'strSearch': 'P',
        'strToDate': to_date,
        'strType': 'C'
    }

    print(f"URL: {url}")
    print(f"Params: {params}")

    try:
        response = requests.get(url, headers=BSE_HEADERS, params=params, timeout=15)
        print(f"\nStatus Code: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        print(f"Response Length: {len(response.text)} chars")
        print(f"\nFirst 500 chars of response:")
        print("-" * 40)
        print(response.text[:500])
        print("-" * 40)

        if response.status_code == 200:
            try:
                data = response.json()
                if 'Table' in data:
                    print(f"\n✓ SUCCESS: Got {len(data.get('Table', []))} announcements")
                else:
                    print(f"\n⚠ WARNING: JSON received but no 'Table' key. Keys: {list(data.keys())}")
            except Exception as e:
                print(f"\n✗ FAILED to parse JSON: {e}")
        else:
            print(f"\n✗ FAILED: HTTP {response.status_code}")

    except Exception as e:
        print(f"\n✗ CONNECTION ERROR: {e}")


def test_nse_api():
    """Test NSE API connectivity."""
    print("\n" + "="*60)
    print("Testing NSE API")
    print("="*60)

    # First, create session and get cookies
    session = requests.Session()
    session.headers.update(NSE_HEADERS)

    print("Step 1: Getting session cookies from NSE homepage...")
    try:
        response = session.get("https://www.nseindia.com", timeout=10)
        print(f"  Homepage Status: {response.status_code}")
        print(f"  Cookies obtained: {len(session.cookies)}")
        for cookie in session.cookies:
            print(f"    - {cookie.name}: {cookie.value[:20]}..." if len(cookie.value) > 20 else f"    - {cookie.name}: {cookie.value}")
    except Exception as e:
        print(f"  ✗ Failed to get homepage: {e}")
        return

    time.sleep(1)  # Rate limit

    print("\nStep 2: Testing market status API (simpler endpoint)...")
    try:
        response = session.get("https://www.nseindia.com/api/marketStatus", timeout=10)
        print(f"  Status Code: {response.status_code}")
        print(f"  Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"  ✓ Got market status data")
            except:
                print(f"  Response preview: {response.text[:200]}")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    time.sleep(1)  # Rate limit

    print("\nStep 3: Testing announcements API...")
    from datetime import datetime, timedelta
    from_date = (datetime.now() - timedelta(days=7)).strftime('%d-%m-%Y')
    to_date = datetime.now().strftime('%d-%m-%Y')

    url = "https://www.nseindia.com/api/corporate-announcements"
    params = {
        'index': 'equities',
        'from_date': from_date,
        'to_date': to_date
    }

    print(f"  URL: {url}")
    print(f"  Params: {params}")

    try:
        response = session.get(url, params=params, timeout=15)
        print(f"\n  Status Code: {response.status_code}")
        print(f"  Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        print(f"  Response Length: {len(response.text)} chars")
        print(f"\n  First 500 chars of response:")
        print("  " + "-" * 38)
        for line in response.text[:500].split('\n'):
            print(f"  {line}")
        print("  " + "-" * 38)

        if response.status_code == 200:
            try:
                data = response.json()
                if isinstance(data, list):
                    print(f"\n  ✓ SUCCESS: Got {len(data)} announcements")
                else:
                    print(f"\n  ⚠ WARNING: Expected list, got {type(data)}")
            except Exception as e:
                print(f"\n  ✗ FAILED to parse JSON: {e}")
        else:
            print(f"\n  ✗ FAILED: HTTP {response.status_code}")

    except Exception as e:
        print(f"\n  ✗ CONNECTION ERROR: {e}")


def main():
    print("="*60)
    print("BSE/NSE API Connectivity Test")
    print("="*60)
    print("\nThis script tests direct API connectivity to BSE and NSE.")
    print("If APIs are blocked, you may need to use a VPN or proxy.\n")

    test_bse_api()
    time.sleep(2)
    test_nse_api()

    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)
    print("""
Troubleshooting Tips:
1. If you see HTML in the response, the API is likely blocked
2. Try using a VPN (the APIs may be geo-restricted)
3. Try from a different network (your IP may be rate-limited)
4. The APIs may have changed - check BSE/NSE websites for updates
5. Consider using an API service like Alpha Vantage for market data
""")


if __name__ == "__main__":
    main()
