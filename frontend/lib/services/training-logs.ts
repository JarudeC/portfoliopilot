// Service for managing training session logs in database
import { createClient, createServiceRoleClient } from '@/lib/supabase/server';
import { createClient as createBrowserClient } from '@/lib/supabase/client';
import { TrainingLog, CreateTrainingLogData, HydratedTrainingLog, LazyTrainingLog, ForecastResult, BacktestResult, ChartConfig } from '@/lib/types/training';
import { StorageService } from './storage';

export class TrainingLogService {
  private supabase;
  private storageService: StorageService;

  constructor(isServer = false) {
    this.storageService = new StorageService(isServer);
    if (isServer) {
      // Server client requires async initialization
      this.supabase = null;
    } else {
      this.supabase = createBrowserClient();
    }
  }

  private async getSupabaseClient() {
    if (!this.supabase) {
      this.supabase = await createClient();
    }
    return this.supabase;
  }

  async createLog(data: CreateTrainingLogData, userId?: string): Promise<TrainingLog> {
    // Use admin client for server-side operations
    const supabase = createServiceRoleClient();

    let finalUserId = userId;

    // Fallback to current user if no userId provided
    if (!finalUserId) {
      const regularSupabase = await this.getSupabaseClient();
      const { data: { user }, error: authError } = await regularSupabase.auth.getUser();

      if (authError || !user) {
        throw new Error('Authentication required to create training log');
      }
      finalUserId = user.id;
    }

    // Generate a unique ID for storage paths
    const logId = crypto.randomUUID();

    // Upload results to storage
    const resultsUpload = await this.storageService.uploadJson(
      finalUserId,
      'results',
      logId,
      data.results
    );

    // Upload charts to storage (if provided)
    let chartsUrl: string | undefined;
    if (data.charts) {
      const chartsUpload = await this.storageService.uploadJson(
        finalUserId,
        'charts',
        logId,
        data.charts
      );
      chartsUrl = chartsUpload.url;
    }

    const insertData = {
      id: logId,
      user_id: finalUserId,
      type: data.type,
      stocks: data.stocks,
      model: data.model,
      parameters: data.parameters,
      results_url: resultsUpload.url,
      charts_url: chartsUrl,
      metrics: data.metrics,
      status: 'completed'
    };

    const { data: result, error } = await supabase
      .from('training_logs')
      .insert(insertData)
      .select()
      .single();

    if (error) {
      // Clean up uploaded files on DB insert failure
      await this.storageService.deleteFile(resultsUpload.url);
      if (chartsUrl) {
        await this.storageService.deleteFile(chartsUrl);
      }

      console.error('Database insert error details:', {
        code: error.code,
        message: error.message,
        details: error.details,
        hint: error.hint
      });
      throw new Error(`Failed to create training log: ${error.message}`);
    }

    return result;
  }

  /**
   * Hydrate a training log by loading results and charts from storage
   */
  async hydrateLog(log: TrainingLog): Promise<HydratedTrainingLog> {
    const results = log.results_url
      ? await this.storageService.downloadJson<ForecastResult | BacktestResult>(log.results_url)
      : null;

    const charts = log.charts_url
      ? (await this.storageService.downloadJson<ChartConfig[]>(log.charts_url)) ?? undefined
      : undefined;

    // Remove URL fields and add actual data
    const { results_url, charts_url, ...rest } = log;

    return {
      ...rest,
      results: results || { predictions: [] }, // Fallback to empty results
      charts,
    };
  }

  /**
   * Hydrate multiple training logs
   */
  async hydrateLogs(logs: TrainingLog[]): Promise<HydratedTrainingLog[]> {
    return Promise.all(logs.map(log => this.hydrateLog(log)));
  }

  /**
   * Convert a training log to lazy format with signed URLs for on-demand fetching
   */
  async toLazyLog(log: TrainingLog): Promise<LazyTrainingLog> {
    const [resultsSignedUrl, chartsSignedUrl] = await Promise.all([
      log.results_url ? this.storageService.getSignedUrl(log.results_url) : null,
      log.charts_url ? this.storageService.getSignedUrl(log.charts_url) : null,
    ]);

    const { results_url, charts_url, ...rest } = log;

    return {
      ...rest,
      results_signed_url: resultsSignedUrl || undefined,
      charts_signed_url: chartsSignedUrl || undefined,
    };
  }

  /**
   * Convert multiple training logs to lazy format
   */
  async toLazyLogs(logs: TrainingLog[]): Promise<LazyTrainingLog[]> {
    return Promise.all(logs.map(log => this.toLazyLog(log)));
  }

  async getUserLogs(userId?: string, limit = 50, offset = 0): Promise<TrainingLog[]> {
    const supabase = await this.getSupabaseClient();
    
    // Get current user to ensure they can only access their own logs
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      throw new Error('Authentication required to fetch training logs');
    }

    // Always filter by current user's ID, ignore passed userId for security
    const { data, error } = await supabase
      .from('training_logs')
      .select('*')
      .eq('user_id', user.id) // RLS will also enforce this, but explicit is better
      .order('created_at', { ascending: false })
      .range(offset, offset + limit - 1);

    if (error) {
      throw new Error(`Failed to fetch training logs: ${error.message}`);
    }

    return data || [];
  }

  async getLogById(id: string): Promise<TrainingLog | null> {
    const supabase = await this.getSupabaseClient();
    
    // Get current user for ownership verification
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      throw new Error('Authentication required to fetch training log');
    }

    const { data, error } = await supabase
      .from('training_logs')
      .select('*')
      .eq('id', id)
      .eq('user_id', user.id) // Ensure user can only access their own logs
      .single();

    if (error) {
      if (error.code === 'PGRST116') {
        return null; // Not found or not owned by user
      }
      throw new Error(`Failed to fetch training log: ${error.message}`);
    }

    return data;
  }

  async deleteLog(id: string): Promise<void> {
    const supabase = await this.getSupabaseClient();
    
    // Get current user for ownership verification
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      throw new Error('Authentication required to delete training log');
    }

    const { error } = await supabase
      .from('training_logs')
      .delete()
      .eq('id', id)
      .eq('user_id', user.id); // Ensure user can only delete their own logs

    if (error) {
      throw new Error(`Failed to delete training log: ${error.message}`);
    }
  }

  async updateLogStatus(id: string, status: 'completed' | 'failed' | 'in_progress'): Promise<void> {
    const supabase = await this.getSupabaseClient();
    
    // Get current user for ownership verification
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      throw new Error('Authentication required to update training log');
    }

    const { error } = await supabase
      .from('training_logs')
      .update({ 
        status
      })
      .eq('id', id)
      .eq('user_id', user.id); // Ensure user can only update their own logs

    if (error) {
      throw new Error(`Failed to update training log status: ${error.message}`);
    }
  }

  async getLogsByType(type: 'forecast' | 'backtest', limit = 20): Promise<TrainingLog[]> {
    const supabase = await this.getSupabaseClient();
    
    // Get current user for ownership verification
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    
    if (authError || !user) {
      throw new Error('Authentication required to fetch training logs');
    }

    const { data, error } = await supabase
      .from('training_logs')
      .select('*')
      .eq('type', type)
      .eq('user_id', user.id) // Ensure user can only access their own logs
      .order('created_at', { ascending: false })
      .limit(limit);

    if (error) {
      throw new Error(`Failed to fetch ${type} logs: ${error.message}`);
    }

    return data || [];
  }
}