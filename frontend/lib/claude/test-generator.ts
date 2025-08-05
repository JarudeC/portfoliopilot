import { generatePortfolioWeights, StockData } from './generator';

// Test data
const testStockData: StockData[] = [
  { symbol: 'AAPL', price: 150, marketCap: 2500000000000, volume: 50000000 },
  { symbol: 'GOOGL', price: 2800, marketCap: 1800000000000, volume: 25000000 },
  { symbol: 'MSFT', price: 300, marketCap: 2200000000000, volume: 30000000 },
];

// Mock Claude API call
const mockClaudeApi = async (prompt: string): Promise<string> => {
  return `
function calculateWeights(stockData: any[]): number[] {
  // Equal weight strategy
  const count = stockData.length;
  return new Array(count).fill(1.0 / count);
}
  `;
};

// Test the generator
async function testGenerator() {
  console.log('Testing Portfolio Weight Generator...\n');
  
  // Test 1: Normal operation with mock Claude API
  console.log('Test 1: Normal operation');
  const result1 = await generatePortfolioWeights(
    'Equal weight strategy', 
    testStockData, 
    mockClaudeApi
  );
  console.log('Success:', result1.success);
  console.log('Weights:', result1.weights);
  console.log('Fallback used:', result1.fallbackUsed);
  console.log('Execution time:', result1.executionTime, 'ms\n');
  
  // Test 2: Without Claude API (fallback mode)
  console.log('Test 2: Fallback mode (no Claude API)');
  const result2 = await generatePortfolioWeights(
    'Market cap weighted strategy', 
    testStockData
  );
  console.log('Success:', result2.success);
  console.log('Weights:', result2.weights);
  console.log('Fallback used:', result2.fallbackUsed);
  console.log('Execution time:', result2.executionTime, 'ms\n');
  
  // Test 3: Empty stock data
  console.log('Test 3: Empty stock data');
  const result3 = await generatePortfolioWeights(
    'Any strategy', 
    []
  );
  console.log('Success:', result3.success);
  console.log('Weights:', result3.weights);
  console.log('Error:', result3.error);
  console.log('Fallback used:', result3.fallbackUsed);
  console.log('Execution time:', result3.executionTime, 'ms\n');
  
  // Test 4: Malformed Claude response
  console.log('Test 4: Malformed Claude response');
  const badClaudeApi = async (prompt: string): Promise<string> => {
    return 'This is not valid TypeScript code at all!';
  };
  const result4 = await generatePortfolioWeights(
    'Any strategy', 
    testStockData, 
    badClaudeApi
  );
  console.log('Success:', result4.success);
  console.log('Weights:', result4.weights);
  console.log('Fallback used:', result4.fallbackUsed);
  console.log('Execution time:', result4.executionTime, 'ms\n');
  
  console.log('All tests completed successfully!');
}

// Only run if this file is executed directly
if (require.main === module) {
  testGenerator().catch(console.error);
}

export { testGenerator };