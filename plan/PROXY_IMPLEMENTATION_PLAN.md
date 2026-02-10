# Proxy Implementation Plan for Forex Data Providers

## I. Problem Statement

**Issue**: Forex data providers (Dukascopy, potentially others) may be geo-blocked in Vietnam.

**Symptoms**:
- Connection timeouts
- HTTP 403 Forbidden responses
- DNS resolution failures
- Slow/failed downloads from forex data endpoints

**Scope**: Need proxy support for all data providers that make HTTP requests:
- DukascopyArchiveProvider (planned)
- BinanceArchiveProvider (existing)
- BinanceKlines (existing)
- Any future HTTP-based providers

---

## II. Solution Architecture

### A. Design Principles

1. **Optional by default**: Proxy should be opt-in, not required
2. **Provider-agnostic**: Single proxy configuration for all providers
3. **Flexible protocols**: Support HTTP, HTTPS, SOCKS5 proxies
4. **Failover**: Support proxy lists with automatic failover
5. **Performance**: Connection pooling, timeout handling
6. **Security**: Support authenticated proxies
7. **Testing**: Easy to test with/without proxy

### B. Configuration Hierarchy

```
Environment Variables (highest priority)
  ↓
Provider constructor arguments
  ↓
Config file (~/.intraday/proxy.json)
  ↓
No proxy (default)
```

### C. Proxy Types to Support

| Type | Use Case | Protocol |
|------|----------|----------|
| HTTP | Simple proxy | http://host:port |
| HTTPS | Encrypted tunnel | https://host:port |
| SOCKS5 | Most flexible | socks5://host:port |
| Authenticated | Paid proxies | http://user:pass@host:port |

---

## III. Implementation Details

### A. Core Proxy Module

**File**: `intraday/proxy.py` (~150 lines)

```python
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
import os
import json
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

@dataclass
class ProxyConfig:
    http: Optional[str] = None
    https: Optional[str] = None
    socks5: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    verify_ssl: bool = True

class ProxyManager:
    def __init__(self, config: Optional[ProxyConfig] = None):
        self.config = config or self._load_config()
        self._session = None
        self._proxy_index = 0
        self._proxy_list = []

    def _load_config(self) -> ProxyConfig:
        # Priority 1: Environment variables
        if os.getenv('INTRADAY_PROXY'):
            return self._parse_env_proxy()

        # Priority 2: Config file
        config_path = Path.home() / '.intraday' / 'proxy.json'
        if config_path.exists():
            return self._parse_config_file(config_path)

        # Priority 3: No proxy
        return ProxyConfig()

    def _parse_env_proxy(self) -> ProxyConfig:
        proxy_url = os.getenv('INTRADAY_PROXY')
        if proxy_url.startswith('socks5://'):
            return ProxyConfig(socks5=proxy_url)
        elif proxy_url.startswith('https://'):
            return ProxyConfig(https=proxy_url)
        else:
            return ProxyConfig(http=proxy_url)

    def _parse_config_file(self, path: Path) -> ProxyConfig:
        with open(path) as f:
            data = json.load(f)
        return ProxyConfig(**data)

    def get_session(self) -> requests.Session:
        if self._session is None:
            self._session = self._create_session()
        return self._session

    def _create_session(self) -> requests.Session:
        session = requests.Session()

        # Setup retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # Setup proxies
        if self.config.http or self.config.https or self.config.socks5:
            session.proxies = self._build_proxy_dict()

        # SSL verification
        session.verify = self.config.verify_ssl

        return session

    def _build_proxy_dict(self) -> Dict[str, str]:
        proxies = {}
        if self.config.http:
            proxies['http'] = self.config.http
        if self.config.https:
            proxies['https'] = self.config.https
        if self.config.socks5:
            # SOCKS5 works for both HTTP and HTTPS
            proxies['http'] = self.config.socks5
            proxies['https'] = self.config.socks5
        return proxies

    def test_connection(self, test_url: str = "https://www.dukascopy.com") -> bool:
        try:
            session = self.get_session()
            response = session.get(test_url, timeout=self.config.timeout)
            return response.status_code == 200
        except Exception:
            return False

    def rotate_proxy(self):
        # For future: support proxy lists with rotation
        pass
```

---

### B. Provider Integration Pattern

**Modification to existing providers**: Minimal changes

**Example - BinanceArchiveProvider** (`intraday/providers/binance.py`):

```python
class BinanceArchiveProvider(Provider):
    def __init__(
        self,
        data_dir: str,
        symbol: str,
        date_from: Union[None, str, arrow.Arrow, datetime] = None,
        date_to: Union[None, str, arrow.Arrow, datetime] = None,
        proxy_config: Optional[ProxyConfig] = None  # NEW
    ):
        # ... existing init code ...
        self.proxy_manager = ProxyManager(proxy_config)  # NEW

        # ... rest of init ...

    @staticmethod
    def download_month_archive(
        symbol: str,
        month: arrow.Arrow,
        file_path_temp: str,
        proxy_manager: ProxyManager = None  # NEW
    ):
        url = f'https://data.binance.vision/data/spot/monthly/aggTrades/{symbol}/{symbol}-aggTrades-{month.format("YYYY-MM")}.zip'

        if proxy_manager:  # NEW
            session = proxy_manager.get_session()
            response = session.get(url, timeout=proxy_manager.config.timeout)
        else:  # OLD (backward compatible)
            response = requests.get(url)

        # ... rest of download logic ...
```

**Key Points**:
- Backward compatible (proxy_manager=None works)
- Minimal code changes per provider
- Reuses session for connection pooling

---

### C. DukascopyArchiveProvider Integration

**File**: `intraday/providers/dukascopy.py` (from DUKASCOPY_INTEGRATION_PLAN.md)

```python
class DukascopyArchiveProvider(Provider):
    def __init__(
        self,
        data_dir: str,
        symbol: str,
        date_from: Union[None, str, arrow.Arrow, datetime] = None,
        date_to: Union[None, str, arrow.Arrow, datetime] = None,
        proxy_config: Optional[ProxyConfig] = None
    ):
        # ... initialization ...
        self.proxy_manager = ProxyManager(proxy_config)

        # Test connection on init
        if not self.proxy_manager.test_connection():
            raise ConnectionError(
                "Cannot reach Dukascopy. If geo-blocked, configure proxy:\n"
                "  export INTRADAY_PROXY='socks5://proxy.example.com:1080'\n"
                "  or create ~/.intraday/proxy.json"
            )

    @staticmethod
    def download_day_archive(
        symbol: str,
        day: date,
        proxy_manager: ProxyManager
    ) -> List[bytes]:
        session = proxy_manager.get_session()
        bi5_files = []

        # Parallel download with proxy
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def download_hour(hour: int) -> bytes:
            url = f"https://www.dukascopy.com/datafeed/{symbol}/{day.year}/{day.month-1:02d}/{day.day:02d}/{hour:02d}h_ticks.bi5"
            response = session.get(url, timeout=proxy_manager.config.timeout)
            response.raise_for_status()
            return response.content

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {executor.submit(download_hour, h): h for h in range(24)}
            for future in as_completed(futures):
                bi5_files.append(future.result())

        return bi5_files
```

---

## IV. Configuration Methods

### Method 1: Environment Variable (Simplest)

```bash
# HTTP proxy
export INTRADAY_PROXY='http://proxy.example.com:8080'

# SOCKS5 proxy (most common for geo-blocking)
export INTRADAY_PROXY='socks5://127.0.0.1:1080'

# Authenticated proxy
export INTRADAY_PROXY='http://username:password@proxy.example.com:8080'

# Run test
python test/test_dukascopy_api.py
```

### Method 2: Config File (Persistent)

**File**: `~/.intraday/proxy.json`

```json
{
  "http": "http://proxy.example.com:8080",
  "https": "https://proxy.example.com:8080",
  "socks5": "socks5://127.0.0.1:1080",
  "timeout": 30,
  "max_retries": 5,
  "verify_ssl": true
}
```

### Method 3: Programmatic (Most Flexible)

```python
from intraday.proxy import ProxyConfig, ProxyManager
from intraday.providers.dukascopy import DukascopyArchiveProvider

proxy_config = ProxyConfig(
    socks5='socks5://127.0.0.1:1080',
    timeout=60,
    max_retries=5
)

provider = DukascopyArchiveProvider(
    data_dir='./data',
    symbol='EURUSD',
    date_from='2024-01-01',
    date_to='2024-01-31',
    proxy_config=proxy_config
)
```

---

## V. Testing Strategy

### A. Unit Tests

**File**: `test/test_proxy.py`

```python
import unittest
from intraday.proxy import ProxyConfig, ProxyManager
import os

class TestProxyManager(unittest.TestCase):
    def test_no_proxy_by_default(self):
        pm = ProxyManager()
        session = pm.get_session()
        assert session.proxies == {}

    def test_env_proxy_http(self):
        os.environ['INTRADAY_PROXY'] = 'http://localhost:8080'
        pm = ProxyManager()
        session = pm.get_session()
        assert 'http' in session.proxies
        del os.environ['INTRADAY_PROXY']

    def test_env_proxy_socks5(self):
        os.environ['INTRADAY_PROXY'] = 'socks5://localhost:1080'
        pm = ProxyManager()
        session = pm.get_session()
        assert session.proxies['http'].startswith('socks5://')
        del os.environ['INTRADAY_PROXY']

    def test_config_file_loading(self):
        # Create temp config, test loading
        pass

    def test_authenticated_proxy(self):
        config = ProxyConfig(http='http://user:pass@proxy:8080')
        pm = ProxyManager(config)
        assert 'user:pass' in pm.get_session().proxies['http']
```

### B. Integration Tests

**File**: `test/test_dukascopy_with_proxy.py`

```python
import os
import unittest
from intraday.proxy import ProxyConfig
from intraday.providers.dukascopy import DukascopyArchiveProvider

class TestDukascopyWithProxy(unittest.TestCase):
    def test_download_with_proxy(self):
        # Skip if no proxy configured
        if not os.getenv('INTRADAY_PROXY'):
            self.skipTest("Set INTRADAY_PROXY to test")

        provider = DukascopyArchiveProvider(
            data_dir='./test_data',
            symbol='EURUSD',
            date_from='2024-01-15',
            date_to='2024-01-15'
        )

        # Should succeed with proxy
        provider.reset()
        trade = next(provider)
        assert trade is not None
```

---

## VI. Proxy Setup Guide for Vietnam

### A. Recommended Solutions

**Option 1: Free SOCKS5 Proxy (Testing Only)**
```bash
# Install Shadowsocks client
pip install shadowsocks

# Use free SOCKS5 server (search "free socks5 proxy" or use services like ProxyScrape)
export INTRADAY_PROXY='socks5://free-proxy-host:1080'
```

**Option 2: SSH Tunnel (If you have VPS)**
```bash
# Create SOCKS5 proxy via SSH (recommended)
ssh -D 1080 -C -N user@your-vps.com &

# Configure intraday to use local tunnel
export INTRADAY_PROXY='socks5://127.0.0.1:1080'
```

**Option 3: Commercial VPN/Proxy Services**
- Bright Data (former Luminati)
- Smartproxy
- Oxylabs
- IPRoyal

Typical cost: $5-50/month

**Option 4: Cloud-Based Proxy (DIY)**
```bash
# Setup Squid proxy on AWS/GCP/DigitalOcean Singapore/US instance
# Configure security groups for Vietnam IP
# Cost: ~$5/month for micro instance
```

### B. Testing Geo-Block Status

**File**: `test/test_geo_block.py`

```python
import requests

def test_dukascopy_direct():
    """Test if Dukascopy is accessible without proxy"""
    url = "https://www.dukascopy.com/datafeed/EURUSD/2024/00/15/10h_ticks.bi5"
    try:
        response = requests.get(url, timeout=10)
        print(f"Direct access: {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Direct access failed: {e}")
        return False

def test_dukascopy_with_proxy(proxy_url):
    """Test if Dukascopy is accessible with proxy"""
    url = "https://www.dukascopy.com/datafeed/EURUSD/2024/00/15/10h_ticks.bi5"
    proxies = {'http': proxy_url, 'https': proxy_url}
    try:
        response = requests.get(url, proxies=proxies, timeout=30)
        print(f"Proxy access ({proxy_url}): {response.status_code}")
        return response.status_code == 200
    except Exception as e:
        print(f"Proxy access failed: {e}")
        return False
```

---

## VII. Implementation Checklist

### Phase 1: Core Proxy Infrastructure (2-3 hours)
- [ ] Create `intraday/proxy.py` with ProxyManager class
- [ ] Implement ProxyConfig dataclass
- [ ] Environment variable parsing
- [ ] Config file loading (~/.intraday/proxy.json)
- [ ] Session creation with retry logic
- [ ] Connection testing method
- [ ] Unit tests for proxy.py

### Phase 2: Provider Integration (2-3 hours)
- [ ] Update Provider base class signature (optional proxy_config)
- [ ] Modify BinanceArchiveProvider to use ProxyManager
- [ ] Modify BinanceKlines to use ProxyManager
- [ ] Modify MoexArchiveProvider to use ProxyManager
- [ ] Ensure backward compatibility (proxy_config=None)

### Phase 3: Dukascopy Integration (3-4 hours)
- [ ] Implement DukascopyArchiveProvider with proxy support
- [ ] Add geo-block detection and helpful error messages
- [ ] Update test/test_dukascopy_api.py to use proxy if configured
- [ ] Test parallel downloads with proxy

### Phase 4: Testing & Documentation (2 hours)
- [ ] Create test/test_proxy.py
- [ ] Create test/test_geo_block.py
- [ ] Test with SSH tunnel (SOCKS5)
- [ ] Test with HTTP proxy
- [ ] Update CLAUDE.md with proxy configuration examples
- [ ] Add troubleshooting guide

---

## VIII. Error Handling

### Common Proxy Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `ConnectionRefusedError` | Proxy not running | Check proxy service status |
| `ProxyError: 407` | Auth required | Add username:password to proxy URL |
| `SSLError` | SSL verification failed | Set verify_ssl=False (testing only) |
| `Timeout` | Slow proxy | Increase timeout or use faster proxy |
| `403 Forbidden` | Proxy IP also blocked | Use proxy from different region |

### Graceful Degradation

```python
def download_with_fallback(url, proxy_manager):
    try:
        # Try with proxy
        session = proxy_manager.get_session()
        return session.get(url, timeout=30)
    except Exception as e:
        # Log proxy failure
        logger.warning(f"Proxy failed: {e}, trying direct connection")

        # Try without proxy
        return requests.get(url, timeout=10)
```

---

## IX. Performance Considerations

### Proxy Performance Impact

| Scenario | Without Proxy | With Proxy | Impact |
|----------|--------------|------------|---------|
| Single request | 50-200ms | 200-500ms | +150-300ms |
| Parallel (24 reqs) | 1-2s | 3-5s | +2-3s |
| Monthly download | 30-60s | 60-120s | 2x slower |

**Optimization Strategies**:
1. Connection pooling (already in ProxyManager)
2. Increase ThreadPoolExecutor workers to compensate for latency
3. Use SOCKS5 (faster than HTTP CONNECT tunnel)
4. Use geographically closer proxy

### Bandwidth Optimization

```python
# Compress responses if proxy supports it
session.headers.update({
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
})
```

---

## X. Security Considerations

### Sensitive Data

- **Never commit proxy credentials** to git
- Use environment variables or ~/.intraday/proxy.json (add to .gitignore)
- Validate proxy URLs before use

### SSL/TLS

```python
# Production: Always verify SSL
verify_ssl = True

# Development/Testing: May disable for self-signed certs
verify_ssl = False  # Only if you trust the proxy
```

### Proxy Trust

- Free proxies: High risk (MITM attacks)
- SSH tunnel: Trusted (encrypted end-to-end)
- Commercial proxies: Medium risk (read provider's terms)
- Self-hosted: Trusted

---

## XI. Alternative Solutions

### Option A: Cloud Function Proxy

Deploy simple proxy as AWS Lambda or Google Cloud Function:

```python
# lambda_function.py
import requests

def lambda_handler(event, context):
    url = event['url']
    response = requests.get(url)
    return {
        'statusCode': 200,
        'body': response.content,
        'headers': dict(response.headers)
    }
```

**Pros**: No persistent proxy needed, pay-per-use
**Cons**: Cold start latency, request size limits

### Option B: Mirror/Cache Service

Host your own Dukascopy data mirror:
1. VPS in non-blocked region downloads data daily
2. Sync to S3/CloudFront
3. Provider reads from mirror instead of Dukascopy

**Pros**: Fastest, no geo-blocking issues
**Cons**: Storage costs, sync complexity

---

## XII. Recommendation

**Immediate Action**:
1. Test geo-block status: Run `test/test_geo_block.py` from Vietnam
2. If blocked: Setup SSH tunnel to non-Vietnam VPS (easiest, most secure)
3. Implement Phase 1 (core proxy infrastructure)
4. Test Dukascopy downloads with proxy
5. Proceed with Phase 2-4 based on results

**Best Practice for Production**:
- Use SOCKS5 via SSH tunnel for development/testing
- Use commercial proxy service for production (reliability)
- Implement failover proxy list for high availability

---

## XIII. Success Criteria

**Phase 1 Complete**:
- [ ] ProxyManager can load config from env/file
- [ ] test_proxy.py passes all tests
- [ ] Can establish connection through test proxy

**Phase 2 Complete**:
- [ ] All existing providers support optional proxy_config
- [ ] Backward compatibility maintained (existing code works)
- [ ] Integration tests pass with/without proxy

**Phase 3 Complete**:
- [ ] test/test_dukascopy_api.py succeeds via proxy from Vietnam
- [ ] DukascopyArchiveProvider downloads full day successfully
- [ ] Parallel downloads work through proxy

**Production Ready**:
- [ ] Tested with 3+ proxy types (HTTP, SOCKS5, authenticated)
- [ ] Error messages guide users to fix configuration
- [ ] Performance acceptable (< 2x slowdown vs direct)
- [ ] Documentation complete

---

## XIV. Timeline Estimate

| Phase | Task | Duration |
|-------|------|----------|
| 0 | Test geo-block status | 30 min |
| 1 | Core proxy infrastructure | 2-3 hours |
| 2 | Provider integration | 2-3 hours |
| 3 | Dukascopy + proxy | 3-4 hours |
| 4 | Testing & docs | 2 hours |
| **Total** | | **9-12 hours** |

**Blockers**:
- Access to working proxy (SSH tunnel or commercial)
- Verification that proxy resolves geo-blocking

---

## XV. Next Steps

1. **Review this plan** - confirm approach
2. **Test geo-block status** - run simple requests.get() to Dukascopy
3. **Setup test proxy** - SSH tunnel or free SOCKS5 for validation
4. **Implement Phase 1** - core ProxyManager class
5. **Test with Dukascopy API** - modify test/test_dukascopy_api.py to use proxy
6. **Proceed to Phase 2-4** - full integration

**Decision Required**:
- Preferred proxy solution? (SSH tunnel recommended for testing)
- Should proxy support be optional or required for forex providers?
- Should we implement proxy rotation/failover in initial version?
