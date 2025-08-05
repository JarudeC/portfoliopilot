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

const DEFAULT_CONFIG: SecurityConfig = {
  enablePromptValidation: true,
  enableCodeValidation: true,
  strictMode: false,
  allowCreativeStrategies: true,
};

// Very restrictive list - only block obvious security threats
const PROMPT_INJECTION_PATTERNS = [
  // Direct prompt injection attempts
  /ignore\s+(previous|all\s+previous|the\s+previous|your\s+previous)\s+(instructions?|prompts?|commands?)/i,
  /instead\s+of\s+(following|doing|executing|running)/i,
  /system\s+prompt/i,
  /forget\s+(everything|all\s+previous|your\s+instructions)/i,
  
  // AI behavior modification attempts
  /act\s+as\s+(a\s+)?(different|another|new)\s+(ai|assistant|bot)/i,
  /pretend\s+to\s+be\s+(a\s+)?(different|another|hacker|malicious)/i,
  /roleplay\s+as\s+(a\s+)?(hacker|attacker|malicious)/i,
  /you\s+are\s+now\s+(a\s+)?(different|evil|malicious)/i,
];

// Focused on actual security threats in code execution
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

// Allow these patterns even if they might trigger false positives
const ALLOWED_FINANCIAL_PATTERNS = [
  /portfolio/i,
  /strategy/i,
  /weight/i,
  /allocation/i,
  /investment/i,
  /stock/i,
  /sector/i,
  /dividend/i,
  /growth/i,
  /value/i,
  /momentum/i,
  /technical/i,
  /fundamental/i,
  /risk/i,
  /return/i,
  /sharpe/i,
  /volatility/i,
  /correlation/i,
  /diversification/i,
];

function validatePrompt(prompt: string, config: SecurityConfig = DEFAULT_CONFIG): PromptValidationResult {
  if (!config.enablePromptValidation) {
    return {
      isValid: true,
      riskLevel: 'none',
      detectedIntent: 'portfolio_strategy'
    };
  }

  const lowerPrompt = prompt.toLowerCase();
  const blockedPatterns: string[] = [];
  
  // Check for prompt injection attempts
  for (const pattern of PROMPT_INJECTION_PATTERNS) {
    if (pattern.test(prompt)) {
      const patternStr = pattern.toString();
      blockedPatterns.push(patternStr);
      
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
  const behaviorModificationPatterns = [
    /act\s+as/i,
    /pretend\s+to\s+be/i,
    /roleplay\s+as/i,
    /you\s+are\s+now/i,
  ];
  
  for (const pattern of behaviorModificationPatterns) {
    if (pattern.test(prompt)) {
      // But allow if it's clearly about portfolio/investment context
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
  
  // Check custom blocked patterns if provided
  if (config.customBlockedPatterns) {
    for (const customPattern of config.customBlockedPatterns) {
      const regex = new RegExp(customPattern, 'i');
      if (regex.test(prompt)) {
        blockedPatterns.push(customPattern);
      }
    }
    
    if (blockedPatterns.length > 0) {
      return {
        isValid: false,
        blockedReason: 'Custom security rule triggered',
        suggestion: 'Your prompt contains patterns that have been specifically blocked. Please rephrase your portfolio strategy description.',
        riskLevel: 'medium',
        blockedPatterns,
        detectedIntent: 'unknown'
      };
    }
  }
  
  // All good - allow creative and unconventional strategies
  return {
    isValid: true,
    riskLevel: 'none',
    detectedIntent: 'portfolio_strategy'
  };
}

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
  const allowedOperations: string[] = [];
  
  // Check for dangerous code patterns
  for (const pattern of DANGEROUS_CODE_PATTERNS) {
    if (pattern.test(code)) {
      dangerousPatterns.push(pattern.toString());
    }
  }
  
  // Identify allowed operations for transparency
  const allowedPatterns = [
    /Math\./g,
    /Array\.prototype\./g,
    /\.map\(/g,
    /\.filter\(/g,
    /\.reduce\(/g,
    /\.sort\(/g,
    /\.find\(/g,
    /\.forEach\(/g,
    /if\s*\(/g,
    /for\s*\(/g,
    /while\s*\(/g,
    /return/g,
    /const\s+/g,
    /let\s+/g,
    /var\s+/g,
  ];
  
  for (const pattern of allowedPatterns) {
    const matches = code.match(pattern);
    if (matches) {
      allowedOperations.push(pattern.toString().replace(/[\/\\]/g, ''));
    }
  }
  
  if (dangerousPatterns.length > 0) {
    let suggestion = 'Your code contains potentially dangerous operations. ';
    
    // Provide specific suggestions based on what was blocked
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
  
  // Check custom patterns if provided
  if (config.customBlockedPatterns) {
    for (const customPattern of config.customBlockedPatterns) {
      const regex = new RegExp(customPattern, 'i');
      if (regex.test(code)) {
        dangerousPatterns.push(customPattern);
      }
    }
    
    if (dangerousPatterns.length > 0) {
      return {
        isValid: false,
        blockedReason: 'Custom code security rule triggered',
        suggestion: 'Your code contains patterns that have been specifically blocked by custom security rules.',
        riskLevel: 'medium',
        blockedPatterns: dangerousPatterns,
        dangerousPatterns,
        allowedOperations
      };
    }
  }
  
  return {
    isValid: true,
    riskLevel: 'none',
    dangerousPatterns: [],
    allowedOperations
  };
}

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

// Helper function to create custom security configurations
export function createSecurityConfig(overrides: Partial<SecurityConfig>): SecurityConfig {
  return { ...DEFAULT_CONFIG, ...overrides };
}

// Helper function to check if a prompt is likely a legitimate portfolio strategy
export function isLikelyPortfolioStrategy(prompt: string): boolean {
  const prompt_lower = prompt.toLowerCase();
  
  // Check for financial/investment keywords
  const financialKeywords = [
    'portfolio', 'strategy', 'investment', 'stock', 'bond', 'allocation',
    'weight', 'diversification', 'risk', 'return', 'dividend', 'growth',
    'value', 'momentum', 'sector', 'market cap', 'volatility', 'sharpe',
    'correlation', 'rebalancing', 'asset', 'equity', 'index', 'etf'
  ];
  
  const keywordCount = financialKeywords.filter(keyword => 
    prompt_lower.includes(keyword)
  ).length;
  
  // If it has 2+ financial keywords, likely legitimate
  return keywordCount >= 2;
}

// Export validation functions
export { validatePrompt, validateCode };