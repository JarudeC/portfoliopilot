/**
 * Security validation for user prompts and generated code.
 * Blocks prompt injection, behavior modification attempts, and dangerous code patterns.
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

/** Dangerous code patterns that could execute harmful operations */
const DANGEROUS_CODE_PATTERNS = [
  // Code execution and eval
  /\beval\s*\(/i,
  /new\s+Function\s*\(/i,
  /Function\s*\(/i,
  /setTimeout\s*\(/i,
  /setInterval\s*\(/i,

  // DOM and browser access
  /document\./i,
  /window\./i,
  /navigator\./i,
  /location\./i,
  /history\./i,

  // Storage access
  /localStorage/i,
  /sessionStorage/i,
  /cookie/i,
  /indexedDB/i,

  // Network access
  /fetch\s*\(/i,
  /XMLHttpRequest/i,
  /WebSocket/i,
  /axios\./i,

  // System/process access
  /require\s*\(/i,
  /import\s*\(/i,
  /process\./i,
  /fs\./i,
  /child_process/i,
  /exec\s*\(/i,
  /spawn\s*\(/i,
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
