# Server Deployment for blog.pebcac.org

This directory contains Docker Compose configuration for hosting the Hugo blog.

## Architecture

```
GitHub Push → Webhook (port 9000) → rebuild.sh → Hugo build → nginx (port 8080)
                                                                    ↓
                                                         Cloudflare Tunnel
                                                                    ↓
                                                         blog.pebcac.org
```

## Setup Instructions

### 1. Clone the repository

```bash
cd ~
git clone https://github.com/pebcac/blog.git hugo-blog
cd hugo-blog
git submodule update --init --recursive
```

### 2. Configure webhook secret

```bash
# Generate a webhook secret
openssl rand -hex 32

# Copy the example and add your secret
cd server
cp hooks.json.example hooks.json
# Edit hooks.json and replace YOUR_WEBHOOK_SECRET_HERE with the generated secret
```

### 3. Initial Hugo build

```bash
cd ~/hugo-blog
docker run --rm -v "$(pwd)":/src klakegg/hugo:ext-alpine --minify
```

### 4. Start containers

```bash
cd ~/hugo-blog/server
docker compose up -d
```

### 5. Configure GitHub Webhook

1. Go to https://github.com/pebcac/blog/settings/hooks
2. Add webhook:
   - Payload URL: `http://YOUR_SERVER_IP:9000/hooks/rebuild-blog`
   - Content type: `application/json`
   - Secret: (use the secret from step 2)
   - Events: Just the push event
3. Save and verify delivery succeeds

### 6. Update Cloudflare Tunnel

Point `blog.pebcac.org` to `http://localhost:8080`

## Troubleshooting

### Check webhook logs
```bash
docker logs hugo-webhook -f
```

### Check nginx logs
```bash
docker logs hugo-web -f
```

### Manual rebuild
```bash
cd ~/hugo-blog
docker run --rm -v "$(pwd)":/src klakegg/hugo:ext-alpine --minify
```

### Restart services
```bash
cd ~/hugo-blog/server
docker compose restart
```
