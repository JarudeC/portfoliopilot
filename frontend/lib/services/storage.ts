// Service for managing Supabase Storage operations for large JSON data
import { createServiceRoleClient } from '@/lib/supabase/server';
import { createClient as createBrowserClient } from '@/lib/supabase/client';

const BUCKET_NAME = 'user-data';

export interface StorageUploadResult {
  url: string; // Storage path (not full URL)
}

export class StorageService {
  private supabase;

  constructor(isServer = false) {
    if (isServer) {
      this.supabase = createServiceRoleClient();
    } else {
      this.supabase = createBrowserClient();
    }
  }

  /**
   * Upload JSON data to Supabase Storage
   * @param userId - User ID for folder organization
   * @param folder - Folder type: 'results' | 'charts' | 'strategies'
   * @param id - Unique ID for the file (e.g., log_id or strategy_id)
   * @param data - JSON data to store
   * @returns Storage path
   */
  async uploadJson(
    userId: string,
    folder: 'results' | 'charts' | 'strategies',
    id: string,
    data: Record<string, any>
  ): Promise<StorageUploadResult> {
    const path = `${userId}/${folder}/${id}.json`;
    const jsonString = JSON.stringify(data);
    const blob = new Blob([jsonString], { type: 'application/json' });

    const { error } = await this.supabase.storage
      .from(BUCKET_NAME)
      .upload(path, blob, {
        contentType: 'application/json',
        upsert: true, // Overwrite if exists
      });

    if (error) {
      console.error('Storage upload error:', error);
      throw new Error(`Failed to upload to storage: ${error.message}`);
    }

    return { url: path };
  }

  /**
   * Download JSON data from Supabase Storage
   * @param path - Storage path returned from upload
   * @returns Parsed JSON data
   */
  async downloadJson<T = Record<string, any>>(path: string): Promise<T | null> {
    if (!path) return null;

    const { data, error } = await this.supabase.storage
      .from(BUCKET_NAME)
      .download(path);

    if (error) {
      // File might not exist, return null instead of throwing
      if (error.message.includes('not found') || error.message.includes('Object not found')) {
        console.warn(`Storage file not found: ${path}`);
        return null;
      }
      console.error('Storage download error:', error);
      throw new Error(`Failed to download from storage: ${error.message}`);
    }

    if (!data) return null;

    const text = await data.text();
    return JSON.parse(text) as T;
  }

  /**
   * Delete a file from Supabase Storage
   * @param path - Storage path to delete
   */
  async deleteFile(path: string): Promise<void> {
    if (!path) return;

    const { error } = await this.supabase.storage
      .from(BUCKET_NAME)
      .remove([path]);

    if (error) {
      // Don't throw if file doesn't exist
      if (!error.message.includes('not found')) {
        console.error('Storage delete error:', error);
        throw new Error(`Failed to delete from storage: ${error.message}`);
      }
    }
  }

  /**
   * Delete multiple files from Supabase Storage
   * @param paths - Array of storage paths to delete
   */
  async deleteFiles(paths: string[]): Promise<void> {
    const validPaths = paths.filter(p => p);
    if (validPaths.length === 0) return;

    const { error } = await this.supabase.storage
      .from(BUCKET_NAME)
      .remove(validPaths);

    if (error) {
      console.error('Storage bulk delete error:', error);
      // Don't throw, just log - some files might not exist
    }
  }

  /**
   * Get a signed URL for temporary access (useful for client-side downloads)
   * @param path - Storage path
   * @param expiresIn - Expiration time in seconds (default 1 hour)
   */
  async getSignedUrl(path: string, expiresIn = 3600): Promise<string | null> {
    if (!path) return null;

    const { data, error } = await this.supabase.storage
      .from(BUCKET_NAME)
      .createSignedUrl(path, expiresIn);

    if (error) {
      console.error('Storage signed URL error:', error);
      return null;
    }

    return data?.signedUrl || null;
  }
}

// Singleton instances
let serverStorageService: StorageService | null = null;
let browserStorageService: StorageService | null = null;

export function getStorageService(isServer = false): StorageService {
  if (isServer) {
    if (!serverStorageService) {
      serverStorageService = new StorageService(true);
    }
    return serverStorageService;
  } else {
    if (!browserStorageService) {
      browserStorageService = new StorageService(false);
    }
    return browserStorageService;
  }
}
