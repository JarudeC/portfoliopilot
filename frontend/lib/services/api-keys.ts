/**
 * Service for managing user API keys in the database.
 *
 * Handles CRUD operations for encrypted API keys stored in Supabase.
 * Keys are encrypted using AES-256-GCM before storage and decrypted
 * only when needed for API calls.
 *
 * Security features:
 * - All keys are encrypted at rest
 * - Row-level security ensures users can only access their own keys
 * - Key previews (e.g., "sk-ant-...xxxx") are stored for display without decryption
 *
 * IMPORTANT: Decryption functions should only be used server-side.
 */

import { createClient } from '@/lib/supabase/server';
import { createClient as createBrowserClient } from '@/lib/supabase/client';
import {
  encryptApiKey,
  decryptApiKey,
  getKeyPreview,
  isValidApiKeyFormat,
} from './encryption';

/**
 * Supported API providers.
 * Currently only Anthropic, but designed for future expansion.
 */
export type ApiProvider = 'anthropic';

/**
 * API key record as stored in the database.
 */
export interface ApiKeyRecord {
  id: string;
  user_id: string;
  provider: ApiProvider;
  key_preview: string;
  created_at: string;
  updated_at: string;
}

/**
 * Service class for managing user API keys.
 *
 * Usage:
 * - Browser: new ApiKeyService() - for checking if key exists, getting preview
 * - Server: new ApiKeyService(true) - for saving/retrieving actual keys
 */
export class ApiKeyService {
  private supabase: ReturnType<typeof createBrowserClient> | null;
  private isServer: boolean;

  constructor(isServer = false) {
    this.isServer = isServer;
    if (isServer) {
      // Server client requires async initialization
      this.supabase = null;
    } else {
      this.supabase = createBrowserClient();
    }
  }

  /**
   * Get the appropriate Supabase client.
   * Lazily initializes server client when needed.
   */
  private async getSupabaseClient() {
    if (!this.supabase) {
      this.supabase = await createClient();
    }
    return this.supabase;
  }

  /**
   * Get the current authenticated user.
   * Throws if not authenticated.
   */
  private async requireAuth() {
    const supabase = await this.getSupabaseClient();
    const { data: { user }, error } = await supabase.auth.getUser();

    if (error || !user) {
      throw new Error('Authentication required to manage API keys');
    }

    return user;
  }

  /**
   * Save or update an API key for the current user.
   *
   * @param provider - The API provider (e.g., 'anthropic')
   * @param apiKey - The plaintext API key to encrypt and store
   * @throws Error if key format is invalid or save fails
   */
  async saveKey(provider: ApiProvider, apiKey: string): Promise<void> {
    // Validate key format before saving
    if (provider === 'anthropic' && !isValidApiKeyFormat(apiKey)) {
      throw new Error(
        'Invalid API key format. Anthropic keys should start with "sk-ant-" and be at least 40 characters.'
      );
    }

    const user = await this.requireAuth();
    const supabase = await this.getSupabaseClient();

    // Encrypt the key and generate preview
    const encryptedKey = encryptApiKey(apiKey);
    const keyPreview = getKeyPreview(apiKey);

    // Upsert: insert or update if already exists
    const { error } = await supabase
      .from('user_api_keys')
      .upsert(
        {
          user_id: user.id,
          provider,
          encrypted_key: encryptedKey,
          key_preview: keyPreview,
          updated_at: new Date().toISOString(),
        },
        {
          onConflict: 'user_id,provider',
        }
      );

    if (error) {
      console.error('Failed to save API key:', error);
      throw new Error(`Failed to save API key: ${error.message}`);
    }
  }

  /**
   * Get the decrypted API key for a provider.
   * Should only be used server-side for making API calls.
   *
   * @param userId - The user's ID
   * @param provider - The API provider
   * @returns The decrypted API key, or null if not found
   */
  async getKey(userId: string, provider: ApiProvider): Promise<string | null> {
    if (!this.isServer) {
      throw new Error('getKey() can only be used server-side');
    }

    const supabase = await this.getSupabaseClient();

    const { data, error } = await supabase
      .from('user_api_keys')
      .select('encrypted_key')
      .eq('user_id', userId)
      .eq('provider', provider)
      .single();

    if (error) {
      if (error.code === 'PGRST116') {
        // Not found
        return null;
      }
      console.error('Failed to get API key:', error);
      throw new Error(`Failed to get API key: ${error.message}`);
    }

    if (!data?.encrypted_key) {
      return null;
    }

    // Decrypt and return
    return decryptApiKey(data.encrypted_key);
  }

  /**
   * Check if the current user has an API key for a provider.
   *
   * @param provider - The API provider
   * @returns true if key exists
   */
  async hasKey(provider: ApiProvider): Promise<boolean> {
    const user = await this.requireAuth();
    const supabase = await this.getSupabaseClient();

    const { data, error } = await supabase
      .from('user_api_keys')
      .select('id')
      .eq('user_id', user.id)
      .eq('provider', provider)
      .single();

    if (error) {
      if (error.code === 'PGRST116') {
        return false;
      }
      throw new Error(`Failed to check API key: ${error.message}`);
    }

    return !!data;
  }

  /**
   * Get the key preview for display purposes.
   * Does not expose the actual key.
   *
   * @param provider - The API provider
   * @returns The key record with preview, or null if not found
   */
  async getKeyInfo(provider: ApiProvider): Promise<ApiKeyRecord | null> {
    const user = await this.requireAuth();
    const supabase = await this.getSupabaseClient();

    const { data, error } = await supabase
      .from('user_api_keys')
      .select('id, user_id, provider, key_preview, created_at, updated_at')
      .eq('user_id', user.id)
      .eq('provider', provider)
      .single();

    if (error) {
      if (error.code === 'PGRST116') {
        return null;
      }
      throw new Error(`Failed to get API key info: ${error.message}`);
    }

    return data as ApiKeyRecord;
  }

  /**
   * Delete the API key for a provider.
   *
   * @param provider - The API provider
   */
  async deleteKey(provider: ApiProvider): Promise<void> {
    const user = await this.requireAuth();
    const supabase = await this.getSupabaseClient();

    const { error } = await supabase
      .from('user_api_keys')
      .delete()
      .eq('user_id', user.id)
      .eq('provider', provider);

    if (error) {
      console.error('Failed to delete API key:', error);
      throw new Error(`Failed to delete API key: ${error.message}`);
    }
  }
}

/**
 * Create a server-side API key service instance.
 * Use this in API routes to retrieve user API keys.
 */
export function createServerApiKeyService(): ApiKeyService {
  return new ApiKeyService(true);
}

/**
 * Create a browser-side API key service instance.
 * Use this in React components to check/manage keys.
 */
export function createBrowserApiKeyService(): ApiKeyService {
  return new ApiKeyService(false);
}
