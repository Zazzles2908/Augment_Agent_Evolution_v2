// Four-Brain Dashboard - Database Service
// Stage 4: Local Supabase Features Integration

import { supabase } from './supabase'

// Types for database entities
export interface UserProfile {
  id: string
  user_id: string
  display_name?: string
  preferences: {
    communication_style: string
    detail_level: string
    formality: string
    response_length: string
  }
  personality_traits: {
    concise_verbose: number
    formal_casual: number
    technical_simple: number
    direct_diplomatic: number
  }
  created_at: string
  updated_at: string
}

export interface Document {
  id: string
  user_id: string
  filename: string
  document_type: string
  processing_status: string
  content_text?: string
  metadata?: any
  created_at: string
  updated_at: string
}

export interface TaskLog {
  id: string
  task_id: string
  user_id: string
  task_type: string
  status: string
  input_data?: any
  output_data?: any
  processing_time_ms?: number
  error_message?: string
  created_at: string
  updated_at: string
  completed_at?: string
}

export interface ChatSession {
  id: string
  user_id: string
  session_title?: string
  session_context?: any
  message_count: number
  created_at: string
  updated_at: string
  last_activity: string
}

export interface ChatMessage {
  id: string
  session_id: string
  user_id: string
  message_type: 'user' | 'assistant' | 'system'
  content: string
  metadata?: any
  task_id?: string
  created_at: string
}

// Database service class
export class DatabaseService {
  
  // User Profile Management
  static async getUserProfile(userId: string): Promise<UserProfile | null> {
    const { data, error } = await supabase
      .schema('augment_agent')
      .from('user_profiles')
      .select('*')
      .eq('user_id', userId)
      .single()
    
    if (error) {
      console.error('Error fetching user profile:', error)
      return null
    }
    
    return data
  }

  static async createOrUpdateUserProfile(profile: Partial<UserProfile>): Promise<UserProfile | null> {
    const { data, error } = await supabase
      .schema('augment_agent')
      .from('user_profiles')
      .upsert({
        user_id: profile.user_id,
        display_name: profile.display_name,
        preferences: profile.preferences,
        personality_traits: profile.personality_traits,
        updated_at: new Date().toISOString()
      })
      .select()
      .single()
    
    if (error) {
      console.error('Error updating user profile:', error)
      return null
    }
    
    return data
  }

  // Document Management
  static async getUserDocuments(userId: string): Promise<Document[]> {
    const { data, error } = await supabase
      .schema('augment_agent')
      .from('documents')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false })
    
    if (error) {
      console.error('Error fetching user documents:', error)
      return []
    }
    
    return data || []
  }

  static async getDocumentById(documentId: string): Promise<Document | null> {
    const { data, error } = await supabase
      .from('documents')
      .select('*')
      .eq('id', documentId)
      .single()
    
    if (error) {
      console.error('Error fetching document:', error)
      return null
    }
    
    return data
  }

  // Task Logging
  static async getUserTaskLogs(userId: string, limit: number = 50): Promise<TaskLog[]> {
    const { data, error } = await supabase
      .from('task_logs')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false })
      .limit(limit)
    
    if (error) {
      console.error('Error fetching task logs:', error)
      return []
    }
    
    return data || []
  }

  static async getTaskLogByTaskId(taskId: string): Promise<TaskLog | null> {
    const { data, error } = await supabase
      .from('task_logs')
      .select('*')
      .eq('task_id', taskId)
      .single()
    
    if (error) {
      console.error('Error fetching task log:', error)
      return null
    }
    
    return data
  }

  // Chat History Management
  static async getUserChatSessions(userId: string): Promise<ChatSession[]> {
    const { data, error } = await supabase
      .from('chat_sessions')
      .select('*')
      .eq('user_id', userId)
      .order('last_activity', { ascending: false })
    
    if (error) {
      console.error('Error fetching chat sessions:', error)
      return []
    }
    
    return data || []
  }

  static async createChatSession(userId: string, title?: string): Promise<ChatSession | null> {
    const { data, error } = await supabase
      .from('chat_sessions')
      .insert({
        user_id: userId,
        session_title: title || `Chat ${new Date().toLocaleDateString()}`,
        session_context: {},
        message_count: 0
      })
      .select()
      .single()
    
    if (error) {
      console.error('Error creating chat session:', error)
      return null
    }
    
    return data
  }

  static async getChatMessages(sessionId: string): Promise<ChatMessage[]> {
    const { data, error } = await supabase
      .from('chat_messages')
      .select('*')
      .eq('session_id', sessionId)
      .order('created_at', { ascending: true })
    
    if (error) {
      console.error('Error fetching chat messages:', error)
      return []
    }
    
    return data || []
  }

  static async addChatMessage(message: Omit<ChatMessage, 'id' | 'created_at'>): Promise<ChatMessage | null> {
    const { data, error } = await supabase
      .from('chat_messages')
      .insert(message)
      .select()
      .single()
    
    if (error) {
      console.error('Error adding chat message:', error)
      return null
    }

    // Update session message count and last activity
    await supabase
      .from('chat_sessions')
      .update({
        message_count: supabase.rpc('increment_message_count', { session_id: message.session_id }),
        last_activity: new Date().toISOString(),
        updated_at: new Date().toISOString()
      })
      .eq('id', message.session_id)
    
    return data
  }

  // Real-time subscriptions
  static subscribeToTaskUpdates(userId: string, callback: (payload: any) => void) {
    return supabase
      .channel('task_updates')
      .on('postgres_changes', {
        event: 'UPDATE',
        schema: 'augment_agent',
        table: 'task_logs',
        filter: `user_id=eq.${userId}`
      }, callback)
      .subscribe()
  }

  static subscribeToNewMessages(sessionId: string, callback: (payload: any) => void) {
    return supabase
      .channel('chat_messages')
      .on('postgres_changes', {
        event: 'INSERT',
        schema: 'augment_agent',
        table: 'chat_messages',
        filter: `session_id=eq.${sessionId}`
      }, callback)
      .subscribe()
  }

  // Knowledge Search
  static async searchKnowledge(userId: string, query: string, limit: number = 10) {
    // This will be enhanced when vector search is fully implemented
    const { data, error } = await supabase
      .from('knowledge')
      .select('*')
      .or(`user_id.eq.${userId},user_id.is.null`)
      .textSearch('knowledge_content', query)
      .limit(limit)
    
    if (error) {
      console.error('Error searching knowledge:', error)
      return []
    }
    
    return data || []
  }
}

// Export default instance
export const db = DatabaseService
