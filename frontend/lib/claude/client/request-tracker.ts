/**
 * Client-side request management for Claude API.
 * Handles request deduplication and rate limiting to prevent API abuse.
 */

import type { GenerateRequest, GenerateResponse } from './client';
import { RATE_LIMIT_WINDOW, MAX_REQUESTS_PER_WINDOW } from '../core/constants';

/**
 * Tracks in-flight requests for deduplication and rate limiting.
 * Prevents duplicate concurrent requests and enforces client-side rate limits.
 */
class RequestTracker {
  private requestCounts: Map<string, { count: number; resetTime: number }> = new Map();
  private pendingRequests: Map<string, Promise<GenerateResponse>> = new Map();

  /**
   * Generate a hash from request parameters for deduplication.
   * Normalizes description and sorts symbols for consistent hashing.
   */
  private generateRequestHash(request: GenerateRequest): string {
    const normalized = {
      description: request.userDescription.trim().toLowerCase(),
      symbols: request.stockData.map(s => s.symbol).sort(),
    };
    return btoa(JSON.stringify(normalized));
  }

  /**
   * Register a new request for deduplication tracking.
   * Automatically cleans up when the request completes.
   */
  registerRequest(request: GenerateRequest, promise: Promise<GenerateResponse>): void {
    const hash = this.generateRequestHash(request);
    this.pendingRequests.set(hash, promise);

    promise.finally(() => {
      this.pendingRequests.delete(hash);
    });
  }

  /**
   * Check if a new request is allowed under rate limits.
   * Returns allowed status and rate limit info.
   */
  checkRateLimit(): { allowed: boolean; remaining: number; resetTime: number } {
    const now = Date.now();
    const key = 'client-rate-limit';
    const existing = this.requestCounts.get(key);

    // Reset window if expired
    if (!existing || now >= existing.resetTime) {
      const resetTime = now + RATE_LIMIT_WINDOW;
      this.requestCounts.set(key, { count: 1, resetTime });
      return { allowed: true, remaining: MAX_REQUESTS_PER_WINDOW - 1, resetTime };
    }

    // Check if limit exceeded
    if (existing.count >= MAX_REQUESTS_PER_WINDOW) {
      return { allowed: false, remaining: 0, resetTime: existing.resetTime };
    }

    // Increment count
    existing.count++;
    this.requestCounts.set(key, existing);
    return {
      allowed: true,
      remaining: MAX_REQUESTS_PER_WINDOW - existing.count,
      resetTime: existing.resetTime
    };
  }
}

/** Singleton instance for global request tracking */
export const requestTracker = new RequestTracker();
