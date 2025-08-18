// Four-Brain Dashboard - Supabase Client Configuration
// Stage 3: Supabase Dashboard UI Development

import { createClient } from '@supabase/supabase-js'

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!

if (!supabaseUrl || !supabaseAnonKey) {
  throw new Error('Missing Supabase environment variables')
}

// Create Supabase client
export const supabase = createClient(supabaseUrl, supabaseAnonKey, {
  auth: {
    autoRefreshToken: true,
    persistSession: true,
    detectSessionInUrl: true
  }
})

// Four-Brain Gateway API configuration
export const FOUR_BRAIN_GATEWAY_URL = process.env.NEXT_PUBLIC_FOUR_BRAIN_GATEWAY_URL!

// API helper functions
export const fourBrainApi = {
  // Chat enhancement endpoint
  chat: async (message: string, context?: string, token?: string) => {
    const response = await fetch(`${FOUR_BRAIN_GATEWAY_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({
        query: message,
        context: context,
        personality_traits: {
          style: 'professional',
          detail_level: 'comprehensive'
        }
      })
    })
    
    if (!response.ok) {
      throw new Error(`Chat API error: ${response.status} ${response.statusText}`)
    }
    
    return response.json()
  },

  // Semantic search endpoint
  search: async (query: string, limit: number = 10, token?: string) => {
    const response = await fetch(`${FOUR_BRAIN_GATEWAY_URL}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({
        query,
        limit,
        filters: {}
      })
    })
    
    if (!response.ok) {
      throw new Error(`Search API error: ${response.status} ${response.statusText}`)
    }
    
    return response.json()
  },

  // Document upload endpoint
  upload: async (fileInfo: any, metadata: any, token?: string) => {
    const response = await fetch(`${FOUR_BRAIN_GATEWAY_URL}/upload`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
      },
      body: JSON.stringify({
        file_info: fileInfo,
        metadata: metadata
      })
    })
    
    if (!response.ok) {
      throw new Error(`Upload API error: ${response.status} ${response.statusText}`)
    }
    
    return response.json()
  },

  // Task status endpoint
  getTaskStatus: async (taskId: string, token?: string) => {
    const response = await fetch(`${FOUR_BRAIN_GATEWAY_URL}/tasks/${taskId}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
      }
    })

    // Handle 404 gracefully - task may not exist yet or has been cleaned up
    if (response.status === 404) {
      return {
        status: 'not_found',
        message: 'Task not found or has been cleaned up',
        taskId
      }
    }

    if (!response.ok) {
      throw new Error(`Task status API error: ${response.status} ${response.statusText}`)
    }

    return response.json()
  }
}

// Types for API responses
export interface ChatResponse {
  task_id: string
  response: string
  confidence: number
  sources: Array<{
    type: string
    title: string
    confidence: number
  }>
  processing_time_ms: number
}

export interface SearchResponse {
  task_id: string
  results: Array<{
    id: string
    title: string
    content: string
    score: number
    metadata: any
  }>
  total_found: number
  processing_time_ms: number
}

export interface UploadResponse {
  task_id: string
  status: string
  message: string
  estimated_completion?: string
}

export interface TaskStatusResponse {
  task_id: string
  status: string
  progress: number
  result?: any
  error?: string
  created_at: string
  updated_at: string
}
