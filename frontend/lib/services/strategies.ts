/**
 * Service for managing user-saved AI strategies.
 * Strategies are stored in Supabase with code in Storage.
 */

import { createServiceRoleClient } from '@/lib/supabase/server';
import { StorageService, getStorageService } from './storage';
import type {
  Strategy,
  HydratedStrategy,
  CreateStrategyInput,
  UpdateStrategyInput,
} from '@/lib/types/strategy';

export class StrategyService {
  private supabase;
  private storageService: StorageService;

  constructor() {
    this.supabase = createServiceRoleClient();
    this.storageService = getStorageService(true);
  }

  /**
   * List all strategies for a user, optionally filtered by mode.
   */
  async listStrategies(
    userId: string,
    mode?: 'backtest' | 'forecast'
  ): Promise<HydratedStrategy[]> {
    let query = this.supabase
      .from('user_strategies')
      .select('*')
      .eq('user_id', userId)
      .order('updated_at', { ascending: false });

    if (mode) {
      query = query.eq('mode', mode);
    }

    const { data, error } = await query;

    if (error) {
      console.error('Error listing strategies:', error);
      throw new Error(`Failed to list strategies: ${error.message}`);
    }

    // Hydrate all strategies with their code
    return this.hydrateStrategies(data || []);
  }

  /**
   * Get a single strategy by ID.
   */
  async getStrategy(id: string): Promise<HydratedStrategy | null> {
    const { data, error } = await this.supabase
      .from('user_strategies')
      .select('*')
      .eq('id', id)
      .single();

    if (error) {
      if (error.code === 'PGRST116') {
        return null; // Not found
      }
      console.error('Error getting strategy:', error);
      throw new Error(`Failed to get strategy: ${error.message}`);
    }

    return this.hydrateStrategy(data);
  }

  /**
   * Save a new strategy.
   */
  async saveStrategy(
    userId: string,
    input: CreateStrategyInput
  ): Promise<HydratedStrategy> {
    // Generate ID for the strategy
    const strategyId = crypto.randomUUID();

    // Upload code to storage
    const codeUpload = await this.storageService.uploadJson(
      userId,
      'strategies',
      strategyId,
      { code: input.code }
    );

    // Insert into database
    const { data, error } = await this.supabase
      .from('user_strategies')
      .insert({
        id: strategyId,
        user_id: userId,
        name: input.name,
        description: input.description || null,
        code_url: codeUpload.url,
        mode: input.mode,
      })
      .select()
      .single();

    if (error) {
      // Clean up storage on DB error
      await this.storageService.deleteFile(codeUpload.url);
      console.error('Error saving strategy:', error);
      throw new Error(`Failed to save strategy: ${error.message}`);
    }

    return {
      ...data,
      code: input.code,
    };
  }

  /**
   * Update an existing strategy.
   */
  async updateStrategy(
    id: string,
    userId: string,
    input: UpdateStrategyInput
  ): Promise<HydratedStrategy> {
    // Get existing strategy to verify ownership and get code_url
    const existing = await this.supabase
      .from('user_strategies')
      .select('*')
      .eq('id', id)
      .eq('user_id', userId)
      .single();

    if (existing.error || !existing.data) {
      throw new Error('Strategy not found or access denied');
    }

    const updates: Record<string, any> = {
      updated_at: new Date().toISOString(),
    };

    if (input.name !== undefined) {
      updates.name = input.name;
    }

    if (input.description !== undefined) {
      updates.description = input.description;
    }

    // If code is being updated, upload new version
    if (input.code !== undefined) {
      await this.storageService.uploadJson(userId, 'strategies', id, {
        code: input.code,
      });
    }

    // Update database
    const { data, error } = await this.supabase
      .from('user_strategies')
      .update(updates)
      .eq('id', id)
      .eq('user_id', userId)
      .select()
      .single();

    if (error) {
      console.error('Error updating strategy:', error);
      throw new Error(`Failed to update strategy: ${error.message}`);
    }

    return this.hydrateStrategy(data);
  }

  /**
   * Delete a strategy.
   */
  async deleteStrategy(id: string, userId: string): Promise<void> {
    // Get strategy to get code_url for cleanup
    const { data: strategy } = await this.supabase
      .from('user_strategies')
      .select('code_url')
      .eq('id', id)
      .eq('user_id', userId)
      .single();

    // Delete from database (triggers will handle storage cleanup if configured)
    const { error } = await this.supabase
      .from('user_strategies')
      .delete()
      .eq('id', id)
      .eq('user_id', userId);

    if (error) {
      console.error('Error deleting strategy:', error);
      throw new Error(`Failed to delete strategy: ${error.message}`);
    }

    // Also manually delete from storage (in case trigger isn't set up)
    if (strategy?.code_url) {
      await this.storageService.deleteFile(strategy.code_url);
    }
  }

  /**
   * Check if a strategy name already exists for a user.
   */
  async nameExists(
    userId: string,
    name: string,
    excludeId?: string
  ): Promise<boolean> {
    let query = this.supabase
      .from('user_strategies')
      .select('id')
      .eq('user_id', userId)
      .eq('name', name);

    if (excludeId) {
      query = query.neq('id', excludeId);
    }

    const { data } = await query;
    return (data?.length || 0) > 0;
  }

  /**
   * Hydrate a single strategy with its code from storage.
   */
  private async hydrateStrategy(strategy: Strategy): Promise<HydratedStrategy> {
    let code = '';

    if (strategy.code_url) {
      const codeData = await this.storageService.downloadJson<{ code: string }>(
        strategy.code_url
      );
      code = codeData?.code || '';
    }

    const { code_url, ...rest } = strategy;
    return {
      ...rest,
      code,
    };
  }

  /**
   * Hydrate multiple strategies with their code from storage.
   */
  private async hydrateStrategies(
    strategies: Strategy[]
  ): Promise<HydratedStrategy[]> {
    return Promise.all(strategies.map((s) => this.hydrateStrategy(s)));
  }
}

// Singleton instance
let strategyService: StrategyService | null = null;

export function getStrategyService(): StrategyService {
  if (!strategyService) {
    strategyService = new StrategyService();
  }
  return strategyService;
}
