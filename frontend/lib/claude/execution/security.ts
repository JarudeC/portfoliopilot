/**
 * Security validation for user prompts and generated code.
 * Blocks prompt injection, behavior modification attempts, and dangerous code patterns.
 *
 * ## Security Architecture (Multi-Layer Defense)
 *
 * This module implements Layer 1 of a multi-layer security system:
 *
 * ### Layer 1: Pattern-Based Validation (this file)
 * - Regex patterns to detect dangerous code constructs
 * - Blocks: eval, Function constructor, prototype access, network calls, etc.
 * - Blocks obfuscation techniques: unicode escapes, fromCharCode, base64
 *
 * ### Layer 2: Frozen Globals (code-sandbox.ts)
 * - All exposed globals are frozen to prevent prototype pollution
 * - Limited API surface: only Math, Date, Array methods, etc.
 *
 * ### Layer 3: Global Shadowing (code-sandbox.ts)
 * - Dangerous globals explicitly set to undefined in execution scope
 * - Prevents access even if somehow referenced
 *
 * ### Layer 4: Strict Mode
 * - "use strict" prevents `this` from leaking global object
 * - Prevents undeclared variable access
 *
 * ## Test Cases for Security Validation
 *
 * The following attacks should be BLOCKED:
 *
 * ```javascript
 * // 1. Prototype chain escape
 * [].constructor.constructor('return this')()
 * // BLOCKED by: /\.constructor\b/i
 *
 * // 2. Proto access
 * obj.__proto__.polluted = true
 * // BLOCKED by: /\.__proto__\b/i
 *
 * // 3. Unicode obfuscation
 * \u0065val('malicious')
 * // BLOCKED by: /\\u[0-9a-fA-F]{4}/
 *
 * // 4. String construction
 * String.fromCharCode(101,118,97,108)
 * // BLOCKED by: /String\.fromCharCode/i
 *
 * // 5. Bracket notation for dangerous props
 * obj['constructor']
 * // BLOCKED by: /\[\s*['"`]constructor['"`]\s*\]/i
 *
 * // 6. Global object access
 * globalThis.fetch(...)
 * // BLOCKED by: /\bglobalThis\b/i
 *
 * // 7. Network requests
 * fetch('/api/secrets')
 * // BLOCKED by: /fetch\s*\(/i
 *
 * // 8. Dynamic code execution
 * eval('malicious')
 * new Function('return this')()
 * // BLOCKED by: /\beval\s*\(/i, /new\s+Function\s*\(/i
 * ```
 *
 * The following LEGITIMATE code should be ALLOWED:
 *
 * ```javascript
 * // Math operations
 * const avg = prices.reduce((a, b) => a + b, 0) / prices.length;
 * const std = Math.sqrt(variance);
 *
 * // Array methods
 * const sorted = [...data].sort((a, b) => b.return - a.return);
 * const filtered = stocks.filter(s => s.volume > 1000000);
 *
 * // Object operations
 * const weights = Object.fromEntries(tickers.map((t, i) => [t, values[i]]));
 *
 * // Date operations
 * const today = new Date();
 * const dayOfWeek = today.getDay();
 * ```
 */

// ============================================================================
// Types
// ============================================================================

export interface ValidationResult {
  isValid: boolean;
  blockedReason?: string;
  suggestion?: string;
  riskLevel: 'none' | 'low' | 'medium' | 'high';
  blockedPatterns?: string[];
}

export interface SecurityConfig {
  enablePromptValidation: boolean;
  enableCodeValidation: boolean;
  strictMode: boolean;
  allowCreativeStrategies: boolean;
  customBlockedPatterns?: string[];
  customAllowedPatterns?: string[];
}

export interface PromptValidationResult extends ValidationResult {
  sanitizedPrompt?: string;
  detectedIntent: 'portfolio_strategy' | 'prompt_injection' | 'behavior_modification' | 'unknown';
}

export interface CodeValidationResult extends ValidationResult {
  sanitizedCode?: string;
  dangerousPatterns: string[];
  allowedOperations: string[];
}

// ============================================================================
// Default Configuration
// ============================================================================

const DEFAULT_CONFIG: SecurityConfig = {
  enablePromptValidation: true,
  enableCodeValidation: true,
  strictMode: false,
  allowCreativeStrategies: true,
};

// ============================================================================
// Security Patterns
// ============================================================================

/** Patterns that detect prompt injection attempts */
const PROMPT_INJECTION_PATTERNS = [
  /ignore\s+(previous|all\s+previous|the\s+previous|your\s+previous)\s+(instructions?|prompts?|commands?)/i,
  /instead\s+of\s+(following|doing|executing|running)/i,
  /system\s+prompt/i,
  /forget\s+(everything|all\s+previous|your\s+instructions)/i,
];

/** Patterns that detect AI behavior modification attempts */
const BEHAVIOR_MODIFICATION_PATTERNS = [
  /act\s+as\s+(a\s+)?(different|another|new)\s+(ai|assistant|bot)/i,
  /pretend\s+to\s+be\s+(a\s+)?(different|another|hacker|malicious)/i,
  /roleplay\s+as\s+(a\s+)?(hacker|attacker|malicious)/i,
  /you\s+are\s+now\s+(a\s+)?(different|evil|malicious)/i,
  /act\s+as/i,
  /pretend\s+to\s+be/i,
  /roleplay\s+as/i,
  /you\s+are\s+now/i,
];

/**
 * Dangerous code patterns that could execute harmful operations.
 *
 * These patterns are organized into categories for clarity.
 * Each pattern blocks a specific attack vector.
 */
const DANGEROUS_CODE_PATTERNS = [
  // =========================================================================
  // CATEGORY 1: Dynamic Code Execution
  // Blocks attempts to execute arbitrary code at runtime
  // =========================================================================
  /\beval\s*\(/i,                    // eval('code')
  /new\s+Function\s*\(/i,            // new Function('code')
  /Function\s*\(/i,                  // Function('code')
  /setTimeout\s*\([^,]*[`'"]/i,      // setTimeout('code', ...) with string
  /setInterval\s*\([^,]*[`'"]/i,     // setInterval('code', ...) with string

  // =========================================================================
  // CATEGORY 2: DOM and Browser Access
  // Blocks access to browser APIs that could manipulate the page
  // =========================================================================
  /\bdocument\b/i,                   // document.anything
  /\bwindow\b/i,                     // window.anything
  /\bnavigator\b/i,                  // navigator.anything
  /\blocation\b/i,                   // location.href, etc.
  /\bhistory\b/i,                    // history.pushState, etc.

  // =========================================================================
  // CATEGORY 3: Storage Access
  // Blocks access to persistent storage mechanisms
  // =========================================================================
  /localStorage/i,                   // localStorage.getItem, etc.
  /sessionStorage/i,                 // sessionStorage.getItem, etc.
  /\bcookie\b/i,                     // document.cookie
  /indexedDB/i,                      // IndexedDB access

  // =========================================================================
  // CATEGORY 4: Network Access
  // Blocks attempts to make network requests
  // =========================================================================
  /\bfetch\s*\(/i,                   // fetch('url')
  /XMLHttpRequest/i,                 // new XMLHttpRequest()
  /WebSocket/i,                      // new WebSocket()
  /\baxios\b/i,                      // axios.get, etc.
  /\bRequest\s*\(/i,                 // new Request()

  // =========================================================================
  // CATEGORY 5: Node.js / System Access
  // Blocks server-side and system-level operations
  // =========================================================================
  /\brequire\s*\(/i,                 // require('module')
  /\bimport\s*\(/i,                  // import('module')
  /\bprocess\b/i,                    // process.env, etc.
  /\bfs\b/i,                         // fs.readFile, etc.
  /child_process/i,                  // child_process.exec
  /\bexec\s*\(/i,                    // exec('command')
  /\bspawn\s*\(/i,                   // spawn('command')

  // =========================================================================
  // CATEGORY 6: Prototype Chain Escape (CRITICAL)
  // Blocks attempts to access global object via prototype chain
  // Example attack: [].constructor.constructor('return this')()
  // =========================================================================
  /\.constructor\b/i,                // obj.constructor
  /\.__proto__\b/i,                  // obj.__proto__
  /\["constructor"\]/i,              // obj["constructor"]
  /\['constructor'\]/i,              // obj['constructor']
  /\[`constructor`\]/i,              // obj[`constructor`]
  /\["__proto__"\]/i,                // obj["__proto__"]
  /\['__proto__'\]/i,                // obj['__proto__']
  /Object\.getPrototypeOf/i,         // Object.getPrototypeOf(obj)
  /Object\.setPrototypeOf/i,         // Object.setPrototypeOf(obj, proto)
  /Object\.getOwnPropertyDescriptor/i, // Can access getters
  /Object\.defineProperty/i,         // Can define malicious properties
  /\bReflect\b/i,                    // Reflect.get, Reflect.construct, etc.

  // =========================================================================
  // CATEGORY 7: Global Object Access
  // Blocks direct access to global scope
  // =========================================================================
  /\bglobalThis\b/i,                 // globalThis.fetch
  /\bself\b/i,                       // self.fetch (web worker global)
  /\bglobal\b/i,                     // global.process (Node.js)

  // =========================================================================
  // CATEGORY 8: Additional Dangerous APIs
  // Blocks other potentially dangerous browser/runtime APIs
  // =========================================================================
  /\bWorker\s*\(/i,                  // new Worker() - can run code in thread
  /SharedArrayBuffer/i,              // Can be used for timing attacks
  /\bProxy\s*\(/i,                   // new Proxy() - can intercept operations
  /\bDebugger\b/i,                   // debugger statement
  /\bwith\s*\(/i,                    // with statement - scope manipulation

  // =========================================================================
  // CATEGORY 9: Obfuscation Detection
  // Blocks common techniques used to hide malicious code
  // =========================================================================

  // Unicode escapes (e.g., \u0065val for "eval")
  /\\u[0-9a-fA-F]{4}/,

  // Character code construction
  /String\.fromCharCode/i,           // String.fromCharCode(101,118,97,108)
  /String\.fromCodePoint/i,          // String.fromCodePoint(...)

  // Base64 encoding (can hide malicious strings)
  /\batob\s*\(/i,                    // atob('base64')
  /\bbtoa\s*\(/i,                    // btoa('string')

  // Bracket notation for sensitive property names
  // This catches obj['eval'], obj['constructor'], etc.
  /\[\s*['"`](?:eval|constructor|__proto__|prototype|window|document|fetch|globalThis|Function|setTimeout|setInterval)['"`]\s*\]/i,
];

/** Financial keywords that indicate legitimate portfolio strategy requests */
const FINANCIAL_KEYWORDS = [
  'portfolio', 'strategy', 'weight', 'allocation', 'investment', 'stock', 'bond',
  'sector', 'dividend', 'growth', 'value', 'momentum', 'technical', 'fundamental',
  'risk', 'return', 'sharpe', 'volatility', 'correlation', 'diversification',
  'market cap', 'rebalancing', 'asset', 'equity', 'index', 'etf'
];

const ALLOWED_FINANCIAL_PATTERNS = FINANCIAL_KEYWORDS.map(keyword => new RegExp(keyword, 'i'));

// ============================================================================
// Validation Functions
// ============================================================================

/**
 * Validate a user prompt for security threats.
 */
function validatePrompt(prompt: string, config: SecurityConfig = DEFAULT_CONFIG): PromptValidationResult {
  if (!config.enablePromptValidation) {
    return {
      isValid: true,
      riskLevel: 'none',
      detectedIntent: 'portfolio_strategy'
    };
  }

  const blockedPatterns: string[] = [];

  // Check for prompt injection attempts
  for (const pattern of PROMPT_INJECTION_PATTERNS) {
    if (pattern.test(prompt)) {
      blockedPatterns.push(pattern.toString());
      return {
        isValid: false,
        blockedReason: 'Prompt injection attempt detected',
        suggestion: 'Please describe your portfolio strategy directly without trying to modify my behavior. For example: "Create a momentum-based strategy focusing on tech stocks" or "Build a dividend-focused portfolio with low volatility".',
        riskLevel: 'high',
        blockedPatterns,
        detectedIntent: 'prompt_injection'
      };
    }
  }

  // Check for AI behavior modification
  for (const pattern of BEHAVIOR_MODIFICATION_PATTERNS) {
    if (pattern.test(prompt)) {
      // Allow if it's clearly about portfolio/investment context
      const hasFinancialContext = ALLOWED_FINANCIAL_PATTERNS.some(fp => fp.test(prompt));
      if (!hasFinancialContext) {
        return {
          isValid: false,
          blockedReason: 'Attempt to modify AI behavior detected',
          suggestion: 'Please focus on describing your investment strategy. For example: "Create a growth strategy" or "Build a conservative portfolio with bonds and dividend stocks".',
          riskLevel: 'medium',
          blockedPatterns: [pattern.toString()],
          detectedIntent: 'behavior_modification'
        };
      }
    }
  }

  // Check custom patterns
  const customValidation = validateCustomPatterns(prompt, config.customBlockedPatterns);
  if (!customValidation.isValid) {
    return {
      isValid: false,
      blockedReason: 'Custom security rule triggered',
      suggestion: 'Your prompt contains patterns that have been specifically blocked. Please rephrase your portfolio strategy description.',
      riskLevel: 'medium',
      blockedPatterns: customValidation.blockedPatterns,
      detectedIntent: 'unknown'
    };
  }

  return {
    isValid: true,
    riskLevel: 'none',
    detectedIntent: 'portfolio_strategy'
  };
}

/**
 * Validate code for dangerous patterns.
 */
function validateCode(code: string, config: SecurityConfig = DEFAULT_CONFIG): CodeValidationResult {
  if (!config.enableCodeValidation) {
    return {
      isValid: true,
      riskLevel: 'none',
      dangerousPatterns: [],
      allowedOperations: []
    };
  }

  const dangerousPatterns: string[] = [];

  // Check for dangerous code patterns
  for (const pattern of DANGEROUS_CODE_PATTERNS) {
    if (pattern.test(code)) {
      dangerousPatterns.push(pattern.toString());
    }
  }

  // Get allowed operations for transparency
  const allowedOperations = getCodeAllowedOperations(code);

  if (dangerousPatterns.length > 0) {
    const suggestion = generateCodeSecuritySuggestion(dangerousPatterns);

    return {
      isValid: false,
      blockedReason: 'Dangerous code patterns detected',
      suggestion,
      riskLevel: 'high',
      blockedPatterns: dangerousPatterns,
      dangerousPatterns,
      allowedOperations
    };
  }

  // Check custom patterns
  const customValidation = validateCustomPatterns(code, config.customBlockedPatterns);
  if (!customValidation.isValid) {
    return {
      isValid: false,
      blockedReason: 'Custom code security rule triggered',
      suggestion: 'Your code contains patterns that have been specifically blocked by custom security rules.',
      riskLevel: 'medium',
      blockedPatterns: customValidation.blockedPatterns,
      dangerousPatterns: customValidation.blockedPatterns,
      allowedOperations
    };
  }

  return {
    isValid: true,
    riskLevel: 'none',
    dangerousPatterns: [],
    allowedOperations
  };
}

/**
 * Main security validation function.
 * Validates both prompt and code, returning combined results.
 */
export function validateSecurity(
  prompt: string,
  code?: string,
  config: SecurityConfig = DEFAULT_CONFIG
): {
  promptValidation: PromptValidationResult;
  codeValidation?: CodeValidationResult;
  overallValid: boolean;
  combinedRiskLevel: 'none' | 'low' | 'medium' | 'high';
} {
  const promptValidation = validatePrompt(prompt, config);
  const codeValidation = code ? validateCode(code, config) : undefined;

  const overallValid = promptValidation.isValid && (codeValidation?.isValid !== false);

  // Determine combined risk level
  const risks = [promptValidation.riskLevel];
  if (codeValidation) {
    risks.push(codeValidation.riskLevel);
  }

  const combinedRiskLevel = risks.includes('high') ? 'high' :
                           risks.includes('medium') ? 'medium' :
                           risks.includes('low') ? 'low' : 'none';

  return {
    promptValidation,
    codeValidation,
    overallValid,
    combinedRiskLevel
  };
}

/**
 * Create a custom security configuration.
 */
export function createSecurityConfig(overrides: Partial<SecurityConfig>): SecurityConfig {
  return { ...DEFAULT_CONFIG, ...overrides };
}

// ============================================================================
// Helper Functions
// ============================================================================

/** Validate against custom patterns */
function validateCustomPatterns(input: string, customPatterns?: string[]): { isValid: boolean; blockedPatterns: string[] } {
  if (!customPatterns || customPatterns.length === 0) {
    return { isValid: true, blockedPatterns: [] };
  }

  const blockedPatterns: string[] = [];

  for (const customPattern of customPatterns) {
    const regex = new RegExp(customPattern, 'i');
    if (regex.test(input)) {
      blockedPatterns.push(customPattern);
    }
  }

  return {
    isValid: blockedPatterns.length === 0,
    blockedPatterns
  };
}

/** Get list of allowed operations found in code */
function getCodeAllowedOperations(code: string): string[] {
  const allowedPatterns = [
    /Math\./g, /Array\.prototype\./g, /\.map\(/g, /\.filter\(/g, /\.reduce\(/g,
    /\.sort\(/g, /\.find\(/g, /\.forEach\(/g, /if\s*\(/g, /for\s*\(/g,
    /while\s*\(/g, /return/g, /const\s+/g, /let\s+/g, /var\s+/g
  ];

  const allowedOperations: string[] = [];
  for (const pattern of allowedPatterns) {
    const matches = code.match(pattern);
    if (matches) {
      allowedOperations.push(pattern.toString().replace(/[\/\\]/g, ''));
    }
  }

  return allowedOperations;
}

/** Generate helpful suggestion based on blocked patterns */
function generateCodeSecuritySuggestion(dangerousPatterns: string[]): string {
  let suggestion = 'Your code contains potentially dangerous operations. ';

  if (dangerousPatterns.some(p => p.includes('eval') || p.includes('Function'))) {
    suggestion += 'Instead of eval() or Function constructor, use direct mathematical operations and conditionals. ';
  }

  if (dangerousPatterns.some(p => p.includes('fetch') || p.includes('XMLHttpRequest'))) {
    suggestion += 'Network requests are not allowed in portfolio calculations. Use the provided stockData parameter instead. ';
  }

  if (dangerousPatterns.some(p => p.includes('localStorage') || p.includes('sessionStorage'))) {
    suggestion += 'Storage access is not allowed. Perform calculations using only the input data. ';
  }

  if (dangerousPatterns.some(p => p.includes('document') || p.includes('window'))) {
    suggestion += 'DOM access is not allowed in portfolio calculations. Focus on pure computational logic. ';
  }

  suggestion += 'You can use: Math operations, array methods (.map, .filter, .reduce), conditionals, loops, and variables.';
  return suggestion;
}

// Export validation functions for direct use
export { validatePrompt, validateCode };
