/**
 * Loading state management for Claude API requests.
 * Provides observable progress tracking for React components.
 */

import { LOADING_CLEANUP_DELAY } from '../core/constants';
import type { ProgressState } from '../core/constants';

/**
 * Loading state for a request in progress.
 */
export interface LoadingState {
  isLoading: boolean;
  requestId?: string;
  startTime?: number;
  progress?: ProgressState;
}

/**
 * Manages loading states for active requests with progress tracking.
 * Components can subscribe to state changes for UI updates.
 */
export class LoadingStateManager {
  private states: Map<string, LoadingState> = new Map();
  private listeners: Map<string, ((state: LoadingState) => void)[]> = new Map();

  /**
   * Create a new loading state for a request.
   */
  createRequest(requestId: string): void {
    const state: LoadingState = {
      isLoading: true,
      requestId,
      startTime: Date.now(),
      progress: 'validating'
    };
    this.states.set(requestId, state);
    this.notifyListeners(requestId, state);
  }

  /**
   * Update the progress stage of a request.
   */
  updateProgress(requestId: string, progress: LoadingState['progress']): void {
    const state = this.states.get(requestId);
    if (state) {
      state.progress = progress;
      this.notifyListeners(requestId, state);
    }
  }

  /**
   * Mark a request as completed and clean up after delay.
   */
  completeRequest(requestId: string): void {
    const state = this.states.get(requestId);
    if (state) {
      state.isLoading = false;
      state.progress = undefined;
      this.notifyListeners(requestId, state);

      // Clean up after a delay to allow final state reads
      setTimeout(() => {
        this.states.delete(requestId);
        this.listeners.delete(requestId);
      }, LOADING_CLEANUP_DELAY);
    }
  }

  /**
   * Get current loading state for a request.
   */
  getState(requestId: string): LoadingState | undefined {
    return this.states.get(requestId);
  }

  /**
   * Subscribe to loading state changes for a request.
   * Returns an unsubscribe function.
   */
  onStateChange(requestId: string, callback: (state: LoadingState) => void): () => void {
    if (!this.listeners.has(requestId)) {
      this.listeners.set(requestId, []);
    }
    this.listeners.get(requestId)!.push(callback);

    // Return unsubscribe function
    return () => {
      const callbacks = this.listeners.get(requestId);
      if (callbacks) {
        const index = callbacks.indexOf(callback);
        if (index > -1) {
          callbacks.splice(index, 1);
        }
      }
    };
  }

  /**
   * Notify all listeners of a state change.
   */
  private notifyListeners(requestId: string, state: LoadingState): void {
    const callbacks = this.listeners.get(requestId);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(state);
        } catch (error) {
          console.error('Error in loading state callback:', error);
        }
      });
    }
  }
}

/** Singleton instance for global loading state management */
export const loadingStateManager = new LoadingStateManager();

/**
 * Helpers for React components to interact with loading states.
 */
export const loadingHelpers = {
  /**
   * Subscribe to loading state changes (use in useEffect).
   */
  useLoadingState: (requestId: string, callback: (state: LoadingState) => void) => {
    return loadingStateManager.onStateChange(requestId, callback);
  },

  /**
   * Get current loading state.
   */
  getLoadingState: (requestId: string) => {
    return loadingStateManager.getState(requestId);
  },
};
