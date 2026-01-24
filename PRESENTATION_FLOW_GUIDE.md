# Portfolio Pilot - Code Flow Guide

## Flow 1: Custom AI Strategy (Before Train)

### Step 1: Algorithm Selection
User selects "Custom AI Strategy" → stores previous algo for revert capability.
- [useForecast.ts:71-84](frontend/app/dashboard/hooks/useForecast.ts#L71-L84)

### Step 2: Popup Opens
`setShowPopup(true)` renders ClaudePopup modal.
- [useForecast.ts:78](frontend/app/dashboard/hooks/useForecast.ts#L78)
- [page.tsx:370-382](frontend/app/dashboard/page.tsx#L370-L382)

### Step 3: User Types Strategy
Textarea with validation (min 50 chars).
- [ClaudePopup.tsx:159-164](frontend/components/claude/ClaudePopup.tsx#L159-L164)

### Step 4: Generate Button Click
Calls `generateCodeOnly` with minimal placeholder data (prices=0, not used in prompt).
- [ClaudePopup.tsx:202-225](frontend/components/claude/ClaudePopup.tsx#L202-L225)

### Step 5: API Route
Frontend calls `/api/claude/generate`.
- [client.ts:generateCodeOnly()](frontend/lib/claude/client/client.ts)

### Step 6: Prompt Creation
Builds rigid prompt with user description injected.
- [code-processing.ts:152-157](frontend/lib/claude/server/code-processing.ts#L152-L157)
- [code-processing.ts:56-102](frontend/lib/claude/server/code-processing.ts#L56-L102) - backtest template
- [code-processing.ts:108-139](frontend/lib/claude/server/code-processing.ts#L108-L139) - forecast template

### Step 7: Code Extraction
Extracts function from Claude response, strips markdown.
- [code-processing.ts:171-220](frontend/lib/claude/server/code-processing.ts#L171-L220)

### Step 8: Security Validation
Checks code for dangerous patterns before returning.
- [generator.ts:97-114](frontend/lib/claude/server/generator.ts#L97-L114)

### Step 9: Code Review Modal
User sees generated code, can edit before approving.
- [ClaudePopup.tsx:495-510](frontend/components/claude/ClaudePopup.tsx#L495-L510)

### Step 10: Code Approved (Not Executed)
Stores code in `claudeStrategy` state. No execution yet.
- [ClaudePopup.tsx:228-250](frontend/components/claude/ClaudePopup.tsx#L228-L250)
- [useForecast.ts:89-109](frontend/app/dashboard/hooks/useForecast.ts#L89-L109)

**Key Point:** Code generation separated from execution - real data fetched only when Train clicked.

---

## Flow 2: Normal Algorithm + Train

### Step 1: Train Click
Calls `runForecast(tickers)`.
- [page.tsx:172](frontend/app/dashboard/page.tsx#L172)

### Step 2: Date Calculation
Converts trading days to calendar days (×1.43 multiplier).
- [useForecast.ts:165-169](frontend/app/dashboard/hooks/useForecast.ts#L165-L169)

### Step 3: Route to Traditional
Non-Claude algos call `runTraditionalForecast`.
- [useForecast.ts:182-185](frontend/app/dashboard/hooks/useForecast.ts#L182-L185)

### Step 4: Backend API Call
POST to `/api/forecast/{algo}` (e.g., `/api/forecast/arima`).
- [useForecast.ts:302-306](frontend/app/dashboard/hooks/useForecast.ts#L302-L306)

### Step 5: Response Processing
Converts to chart series format.
- [useForecast.ts:316-323](frontend/app/dashboard/hooks/useForecast.ts#L316-L323)

### Step 6: Metrics Calculation
Computes MSE/MAE per stock and overall.
- [useForecast.ts:374-389](frontend/app/dashboard/hooks/useForecast.ts#L374-L389)

### Step 7: Database Log
POST to `/api/forecast/log` with full payload.
- [useForecast.ts:443-477](frontend/app/dashboard/hooks/useForecast.ts#L443-L477)

---

## Flow 2B: Custom AI + Train

### Step 1: Check Strategy Exists
Validates `claudeStrategy.code` exists.
- [useForecast.ts:212-216](frontend/app/dashboard/hooks/useForecast.ts#L212-L216)

### Step 2: Fetch Real Prices
GET from `/api/prices` for each ticker.
- [useForecast.ts:224-232](frontend/app/dashboard/hooks/useForecast.ts#L224-L232)

### Step 3: Build Real stockData
Creates object with real lookbackPrices/lookbackDates.
- [useForecast.ts:240-245](frontend/app/dashboard/hooks/useForecast.ts#L240-L245)

### Step 4: Execute AI Code
`executeUserCode()` runs strategy with real data.
- [useForecast.ts:248-257](frontend/app/dashboard/hooks/useForecast.ts#L248-L257)
- [generator.ts:291-382](frontend/lib/claude/server/generator.ts#L291-L382) - execution with security

### Step 5: Handle Predictions
Converts multipliers to absolute prices if needed.
- [useForecast.ts:260-272](frontend/app/dashboard/hooks/useForecast.ts#L260-L272)

---

## Key Design Decisions

1. **Prices=0 during generation** - [ClaudePopup.tsx:206](frontend/components/claude/ClaudePopup.tsx#L206) - Prompt doesn't use prices
2. **Multi-layer fallback** - [generator.ts:147-173](frontend/lib/claude/server/generator.ts#L147-L173) - Never crashes
3. **Security before execution** - [generator.ts:314-332](frontend/lib/claude/server/generator.ts#L314-L332) - Validates code
4. **Revert on cancel** - [useForecast.ts:129-134](frontend/app/dashboard/hooks/useForecast.ts#L129-L134) - Uses `prevAlgo`
