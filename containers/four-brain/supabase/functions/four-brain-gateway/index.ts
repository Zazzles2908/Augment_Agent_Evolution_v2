// Four-Brain System v2 - Supabase Edge Function Gateway
// External Access Configuration: Public Tunnel Integration
// MCP Gateway for authenticated access to local K2-Hub orchestration

import { serve } from "https://deno.land/std@0.168.0/http/server.ts"
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

// External Access Configuration - Public Tunnel URL
const FOUR_BRAIN_TUNNEL_URL = Deno.env.get('FOUR_BRAIN_TUNNEL_URL') || 'https://YOUR_PUBLIC_IP:443'
const SUPABASE_URL = Deno.env.get('SUPABASE_URL')!
const SUPABASE_ANON_KEY = Deno.env.get('SUPABASE_ANON_KEY')!

// Initialize Supabase client
const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY)

// CORS headers
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
}

// Authentication middleware
async function authenticateUser(request: Request) {
  const authHeader = request.headers.get('Authorization')
  if (!authHeader) {
    throw new Error('Missing Authorization header')
  }

  const token = authHeader.replace('Bearer ', '')
  const { data: { user }, error } = await supabase.auth.getUser(token)
  
  if (error || !user) {
    throw new Error('Invalid or expired token')
  }
  
  return user
}

// Route mapping for Four-Brain endpoints
const routeMapping = {
  '/chat': {
    method: 'POST',
    endpoint: '/api/v1/chat-enhance',
    description: 'Enhanced chat interaction with Brain-3 (Zazzles's Agent Intelligence)'
  },
  '/search': {
    method: 'POST', 
    endpoint: '/api/v1/semantic-search',
    description: 'Semantic search via Brain-1 (Embedding) and Brain-2 (Reranking)'
  },
  '/upload': {
    method: 'POST',
    endpoint: '/api/v1/process-document', 
    description: 'Document processing via Brain-4 (Docling) and Brain-1 (Embedding)'
  },
  '/tasks': {
    method: 'GET',
    endpoint: '/api/v1/tasks',
    description: 'Task status monitoring and management'
  }
}

// Forward request to Four-Brain tunnel via public URL
async function forwardToFourBrain(endpoint: string, method: string, body?: any, headers?: Record<string, string>) {
  const url = `${FOUR_BRAIN_TUNNEL_URL}${endpoint}`

  const requestOptions: RequestInit = {
    method,
    headers: {
      'Content-Type': 'application/json',
      'User-Agent': 'Four-Brain-Gateway/2.1-External-Access',
      'X-Gateway-Source': 'Supabase-Edge-Function',
      ...headers
    }
  }

  if (body && (method === 'POST' || method === 'PUT')) {
    requestOptions.body = JSON.stringify(body)
  }

  console.log(`Forwarding ${method} request to Four-Brain tunnel: ${url}`)

  try {
    const response = await fetch(url, requestOptions)

    if (!response.ok) {
      console.error(`Four-Brain tunnel responded with status: ${response.status}`)
      throw new Error(`Tunnel error: ${response.status} ${response.statusText}`)
    }

    const responseData = await response.json()

    return {
      status: response.status,
      data: responseData,
      headers: Object.fromEntries(response.headers.entries())
    }
  } catch (error) {
    console.error('Error connecting to Four-Brain tunnel:', error)

    // Provide detailed error information for troubleshooting
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error(`Network error: Cannot reach Four-Brain tunnel at ${FOUR_BRAIN_TUNNEL_URL}. Check tunnel configuration and connectivity.`)
    }

    throw new Error(`Failed to connect to Four-Brain system: ${error.message}`)
  }
}

// Main handler
serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Extract path from URL
    const url = new URL(req.url)
    const path = url.pathname.replace('/four-brain-gateway', '')

    // Debug logging
    console.log(`DEBUG: Full URL: ${req.url}`)
    console.log(`DEBUG: URL pathname: ${url.pathname}`)
    console.log(`DEBUG: Extracted path: "${path}"`)
    console.log(`DEBUG: Available routes: ${JSON.stringify(Object.keys(routeMapping))}`)

    // Handle root path - return available routes
    if (path === '/' || path === '') {
      return new Response(
        JSON.stringify({
          service: 'Four-Brain System v2 Gateway',
          version: '2.1.0-External-Access',
          description: 'External access gateway to Four-Brain orchestration system via public tunnel',
          available_routes: routeMapping,
          tunnel_url: FOUR_BRAIN_TUNNEL_URL,
          architecture: 'Public Tunnel Access',
          status: 'operational'
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 200
        }
      )
    }

    // Authenticate user
    const user = await authenticateUser(req)
    console.log(`Authenticated user: ${user.email} (${user.id})`)

    // Parse request body if present
    let requestBody = null
    if (req.method === 'POST' || req.method === 'PUT') {
      try {
        requestBody = await req.json()
      } catch (error) {
        console.log('No JSON body or invalid JSON')
      }
    }

    // Route handling
    let targetEndpoint: string
    let targetMethod: string

    if (path.startsWith('/tasks/')) {
      // Handle task status requests: /tasks/{task_id}
      targetEndpoint = `/api/v1${path}`
      targetMethod = 'GET'
    } else if (routeMapping[path]) {
      // Handle mapped routes
      const route = routeMapping[path]
      targetEndpoint = route.endpoint
      targetMethod = route.method
    } else {
      return new Response(
        JSON.stringify({
          error: 'Route not found',
          available_routes: Object.keys(routeMapping),
          requested_path: path,
          debug_info: {
            full_url: req.url,
            pathname: url.pathname,
            extracted_path: path,
            path_length: path.length,
            path_chars: path.split('').map(c => c.charCodeAt(0))
          }
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 404
        }
      )
    }

    // Validate HTTP method
    if (req.method !== targetMethod) {
      return new Response(
        JSON.stringify({
          error: `Method ${req.method} not allowed for ${path}`,
          expected_method: targetMethod
        }),
        {
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 405
        }
      )
    }

    // Add user context to request
    const enhancedBody = requestBody ? {
      ...requestBody,
      user_context: {
        user_id: user.id,
        email: user.email,
        timestamp: new Date().toISOString()
      }
    } : null

    // Forward to Four-Brain tunnel system
    const result = await forwardToFourBrain(
      targetEndpoint,
      targetMethod,
      enhancedBody,
      {
        'X-User-ID': user.id,
        'X-User-Email': user.email || '',
        'X-Gateway': 'Supabase-Edge-Function',
        'X-Request-ID': crypto.randomUUID()
      }
    )

    // Return response
    return new Response(
      JSON.stringify(result.data),
      {
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
          'X-Four-Brain-Status': result.status.toString(),
          'X-User-ID': user.id
        },
        status: result.status
      }
    )

  } catch (error) {
    console.error('Gateway error:', error)
    
    return new Response(
      JSON.stringify({
        error: error.message,
        timestamp: new Date().toISOString(),
        service: 'Four-Brain Gateway'
      }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: error.message.includes('Authorization') ? 401 : 500
      }
    )
  }
})

/* External Access Edge Function Configuration:
 *
 * Environment Variables Required:
 * - FOUR_BRAIN_TUNNEL_URL: Public tunnel URL (router forwarding or Cloudflare Tunnel)
 * - SUPABASE_URL: Your Supabase project URL
 * - SUPABASE_ANON_KEY: Your Supabase anon key
 *
 * Architecture: Public Tunnel Access
 * - Edge Function makes HTTP calls to public tunnel URL
 * - Requires either router port forwarding or Cloudflare Tunnel
 * - Leverages external tunnel for secure access to local system
 *
 * Deploy with:
 * supabase functions deploy four-brain-gateway
 *
 * Usage Examples:
 * POST /functions/v1/four-brain-gateway/chat
 * POST /functions/v1/four-brain-gateway/search
 * POST /functions/v1/four-brain-gateway/upload
 * GET  /functions/v1/four-brain-gateway/tasks/{task_id}
 *
 * Tunnel Options:
 * Option A: Router Port Forwarding
 * - Configure router to forward port 443 to 192.168.0.138:443
 * - Set FOUR_BRAIN_TUNNEL_URL=https://YOUR_PUBLIC_IP:443
 *
 * Option B: Cloudflare Tunnel
 * - Install cloudflared and create tunnel to localhost:443
 * - Set FOUR_BRAIN_TUNNEL_URL=https://your-tunnel.trycloudflare.com
 */
