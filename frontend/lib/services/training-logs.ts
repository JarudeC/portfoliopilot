// Service for managing training session logs in database
import { createClient, createServiceRoleClient } from '@/lib/supabase/server';
import { createClient as createBrowserClient } from '@/lib/supabase/client';
import { TrainingLog, CreateTrainingLogData } from '@/lib/types/training';

export class TrainingLogService {
  private supabase;

  constructor(isServer = false) {
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

    const insertData = {
      user_id: finalUserId,
      type: data.type,
      stocks: data.stocks,
      model: data.model,
      parameters: data.parameters,
      results: data.results,
      charts: data.charts,
      metrics: data.metrics,
      status: 'completed'
    };
    
    const { data: result, error } = await supabase
      .from('training_logs')
      .insert(insertData)
      .select()
      .single();

    if (error) {
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