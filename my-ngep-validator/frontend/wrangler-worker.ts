export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      // Health check
      if (path === '/') {
        return new Response(JSON.stringify({
          status: 'ok',
          service: 'NGEP Validator (Cloudflare Workers)',
          endpoints: [
            'POST /api/validate - Run validation pipeline',
            'POST /api/infer - Run inference',
            'POST /api/upload - Upload model',
            'GET /api/models - List models'
          ]
        }), {
          headers: { 'Content-Type': 'application/json', ...corsHeaders }
        });
      }

      // List models from R2
      if (path === '/api/models' && request.method === 'GET') {
        try {
          const list = await env.MODELS_BUCKET.list({ prefix: 'models/' });
          const models = list.objects.map(obj => ({
            name: obj.key.replace('models/', ''),
            size: obj.size,
            uploaded: obj.uploaded
          }));
          
          return new Response(JSON.stringify({ models }), {
            headers: { 'Content-Type': 'application/json', ...corsHeaders }
          });
        } catch (e) {
          return new Response(JSON.stringify({ models: [] }), {
            headers: { 'Content-Type': 'application/json', ...corsHeaders }
          });
        }
      }

      // Upload model
      if (path === '/api/upload' && request.method === 'POST') {
        const formData = await request.formData();
        const file = formData.get('model');
        
        if (!file) {
          return new Response(JSON.stringify({ error: 'No file provided' }), { status: 400, headers: corsHeaders });
        }

        const buffer = await file.arrayBuffer();
        const key = `models/${file.name}`;
        
        await env.MODELS_BUCKET.put(key, buffer, {
          httpMetadata: { contentType: file.type }
        });

        return new Response(JSON.stringify({ 
          success: true, 
          message: `Model ${file.name} uploaded`,
          key: key
        }), {
          headers: { 'Content-Type': 'application/json', ...corsHeaders }
        });
      }

      // Validation endpoint (demo)
      if (path === '/api/validate' && request.method === 'POST') {
        const { neuronCount, genes, model } = await request.json();

        return new Response(JSON.stringify({
          stage: 'complete',
          neuronCount: neuronCount || 100,
          genes: genes || ['Pvalb', 'VIP', 'SST'],
          message: 'Validation data ready'
        }), {
          headers: { 'Content-Type': 'application/json', ...corsHeaders }
        });
      }

      // Inference endpoint (demo)
      if (path === '/api/infer' && request.method === 'POST') {
        const { model, neuronCount } = await request.json();

        return new Response(JSON.stringify({
          stage: 'complete',
          metrics: {
            Pvalb: { r2: 0.356, rmse: 0.691, mae: 0.552, pearson_r: 0.605 },
            VIP: { r2: 0.298, rmse: 0.742, mae: 0.568, pearson_r: 0.562 },
            SST: { r2: 0.312, rmse: 0.715, mae: 0.563, pearson_r: 0.578 }
          }
        }), {
          headers: { 'Content-Type': 'application/json', ...corsHeaders }
        });
      }

      return new Response(JSON.stringify({ error: 'Not found' }), {
        status: 404,
        headers: { 'Content-Type': 'application/json', ...corsHeaders }
      });

    } catch (error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
        headers: { 'Content-Type': 'application/json', ...corsHeaders }
      });
    }
  }
};
