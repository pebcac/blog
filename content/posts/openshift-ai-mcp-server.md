+++
title = "Building an MCP Server with OpenShift AI"
author = ["Preston Davis"]
description = "A hands-on guide to building a Model Context Protocol (MCP) server using OpenShift AI for scalable LLM deployments"
date = 2026-02-04
tags = ["openshift", "ai", "mcp", "python", "llm"]
categories = ["AI", "DevOps"]
draft = true
toc = true
+++

Large Language Models are revolutionizing how we build applications, but deploying and scaling them remains a challenge. In this post, I'll walk you through building an MCP (Model Context Protocol) server using OpenShift AI—combining the power of enterprise Kubernetes with cutting-edge AI infrastructure.

{{< figure src="/images/openshift-ai-dashboard.png" caption="<span class=\"figure-number\">Figure 1: </span>OpenShift AI Dashboard showing model deployments" width="800px" >}}


## What is MCP? {#what-is-mcp}

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open standard developed by Anthropic that enables seamless communication between AI applications and external tools, data sources, and services. Think of it as a universal adapter for AI—allowing your LLM to interact with databases, APIs, file systems, and more through a standardized interface.

MCP follows a client-server architecture:

-   **MCP Hosts**: Applications like Claude Desktop, IDEs, or custom apps that connect to MCP servers
-   **MCP Servers**: Lightweight services that expose specific capabilities (tools, resources, prompts)
-   **Transport Layer**: Communication via stdio (local) or HTTP with Server-Sent Events (remote)


## Why OpenShift AI? {#why-openshift-ai}

[OpenShift AI](https://www.redhat.com/en/technologies/cloud-computing/openshift/openshift-ai) provides a robust platform for deploying and managing AI/ML workloads:

-   **GPU Scheduling**: Intelligent allocation of NVIDIA GPUs across workloads
-   **Model Serving**: Deploy models with KServe or custom runtimes
-   **Scalability**: Auto-scale based on inference demand
-   **Security**: Enterprise-grade access control and network policies
-   **MLOps Integration**: Pipelines, experiment tracking, and model registry

Combining MCP with OpenShift AI gives you a production-ready system for exposing LLM capabilities as standardized services.


## Prerequisites {#prerequisites}

Before we begin, ensure you have:

1.  Access to an OpenShift cluster with OpenShift AI operator installed
2.  `oc` CLI configured and authenticated
3.  Python 3.11+ installed locally
4.  Basic familiarity with Kubernetes concepts

<!--listend-->

```bash
# Verify OpenShift connection
oc whoami
oc get pods -n redhat-ods-applications

# Check OpenShift AI operator status
oc get csv -n redhat-ods-operator | grep rhods
```


## Project Architecture {#project-architecture}

Our MCP server will provide three key capabilities:

1.  **Tools**: Execute functions like querying a knowledge base or running calculations
2.  **Resources**: Expose data from OpenShift (pods, deployments, logs)
3.  **Prompts**: Pre-defined prompt templates for common tasks

<!--listend-->

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   MCP Client    │────▶│   MCP Server    │────▶│  OpenShift AI   │
│ (Claude Desktop)│     │   (Python)      │     │  (LLM Runtime)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │   OpenShift     │
                        │   Resources     │
                        └─────────────────┘
```


## Step 1: Setting Up the OpenShift AI Project {#step-1-setting-up-the-openshift-ai-project}

First, create a new project for our MCP server deployment:

```bash
# Create project
oc new-project mcp-server-demo

# Add necessary permissions for OpenShift AI
oc adm policy add-role-to-user view -z default -n mcp-server-demo
```


## Step 2: Building the MCP Server {#step-2-building-the-mcp-server}

Create a new Python project with the MCP SDK:

```bash
# Create project directory
mkdir openshift-mcp-server && cd openshift-mcp-server

# Initialize with uv (recommended) or pip
uv init
uv add "mcp[cli]" httpx kubernetes openshift-client
```

Now let's create the core MCP server. Create `server.py`:

```python
#!/usr/bin/env python3
"""OpenShift AI MCP Server - Expose OpenShift resources to LLMs."""

import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
    Resource,
    Prompt,
    PromptArgument,
    PromptMessage,
    GetPromptResult,
)

# Initialize server
server = Server("openshift-mcp")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Tools - Functions the LLM can execute
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools for the LLM."""
    return [
        Tool(
            name="get_pods",
            description="List pods in a namespace with their status",
            inputSchema={
                "type": "object",
                "properties": {
                    "namespace": {
                        "type": "string",
                        "description": "Kubernetes namespace to query"
                    },
                    "label_selector": {
                        "type": "string",
                        "description": "Optional label selector (e.g., 'app=nginx')"
                    }
                },
                "required": ["namespace"]
            }
        ),
        Tool(
            name="get_model_status",
            description="Check the status of an OpenShift AI model deployment",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_name": {
                        "type": "string",
                        "description": "Name of the deployed model"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Namespace where model is deployed"
                    }
                },
                "required": ["model_name", "namespace"]
            }
        ),
        Tool(
            name="query_inference",
            description="Send an inference request to a deployed model",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_endpoint": {
                        "type": "string",
                        "description": "URL of the model inference endpoint"
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Prompt to send to the model"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens in response",
                        "default": 256
                    }
                },
                "required": ["model_endpoint", "prompt"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Execute a tool and return results."""

    if name == "get_pods":
        return await _get_pods(
            arguments["namespace"],
            arguments.get("label_selector")
        )

    elif name == "get_model_status":
        return await _get_model_status(
            arguments["model_name"],
            arguments["namespace"]
        )

    elif name == "query_inference":
        return await _query_inference(
            arguments["model_endpoint"],
            arguments["prompt"],
            arguments.get("max_tokens", 256)
        )

    raise ValueError(f"Unknown tool: {name}")


async def _get_pods(namespace: str, label_selector: str | None) -> list[TextContent]:
    """Fetch pods from OpenShift."""
    import openshift_client as oc

    with oc.project(namespace):
        selector = oc.selector("pods")
        if label_selector:
            selector = selector.narrow(label_selector)

        pods = []
        for pod in selector.objects():
            pods.append({
                "name": pod.name(),
                "status": pod.model.status.phase,
                "ready": _is_pod_ready(pod),
                "restarts": _get_restart_count(pod)
            })

    return [TextContent(
        type="text",
        text=json.dumps(pods, indent=2)
    )]


async def _get_model_status(model_name: str, namespace: str) -> list[TextContent]:
    """Check OpenShift AI InferenceService status."""
    import openshift_client as oc

    with oc.project(namespace):
        isvc = oc.selector(f"inferenceservice/{model_name}").object()

        status = {
            "name": model_name,
            "ready": isvc.model.status.get("conditions", [{}])[0].get("status") == "True",
            "url": isvc.model.status.get("url", "Not available"),
            "predictor": {
                "replicas": isvc.model.status.get("components", {}).get("predictor", {}).get("replicas", 0)
            }
        }

    return [TextContent(
        type="text",
        text=json.dumps(status, indent=2)
    )]


async def _query_inference(endpoint: str, prompt: str, max_tokens: int) -> list[TextContent]:
    """Send inference request to model endpoint."""
    import httpx

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{endpoint}/v1/completions",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
        )
        response.raise_for_status()
        result = response.json()

    return [TextContent(
        type="text",
        text=result["choices"][0]["text"]
    )]


def _is_pod_ready(pod) -> bool:
    """Check if pod is ready."""
    conditions = pod.model.status.get("conditions", [])
    for cond in conditions:
        if cond.get("type") == "Ready":
            return cond.get("status") == "True"
    return False


def _get_restart_count(pod) -> int:
    """Get total restart count for pod."""
    containers = pod.model.status.get("containerStatuses", [])
    return sum(c.get("restartCount", 0) for c in containers)


# ============================================================================
# Resources - Data the LLM can read
# ============================================================================

@server.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources."""
    return [
        Resource(
            uri="openshift://namespaces",
            name="OpenShift Namespaces",
            description="List of all accessible namespaces",
            mimeType="application/json"
        ),
        Resource(
            uri="openshift://models",
            name="Deployed Models",
            description="List of models deployed in OpenShift AI",
            mimeType="application/json"
        )
    ]


@server.read_resource()
async def read_resource(uri: str) -> str:
    """Read a resource by URI."""
    import openshift_client as oc

    if uri == "openshift://namespaces":
        namespaces = [ns.name() for ns in oc.selector("namespaces").objects()]
        return json.dumps(namespaces, indent=2)

    elif uri == "openshift://models":
        models = []
        # Query all InferenceServices across accessible namespaces
        for isvc in oc.selector("inferenceservices").objects():
            models.append({
                "name": isvc.name(),
                "namespace": isvc.namespace(),
                "url": isvc.model.status.get("url", "N/A")
            })
        return json.dumps(models, indent=2)

    raise ValueError(f"Unknown resource: {uri}")


# ============================================================================
# Prompts - Reusable prompt templates
# ============================================================================

@server.list_prompts()
async def list_prompts() -> list[Prompt]:
    """List available prompt templates."""
    return [
        Prompt(
            name="debug_deployment",
            description="Help debug a failing deployment",
            arguments=[
                PromptArgument(
                    name="namespace",
                    description="Namespace containing the deployment",
                    required=True
                ),
                PromptArgument(
                    name="deployment_name",
                    description="Name of the failing deployment",
                    required=True
                )
            ]
        ),
        Prompt(
            name="optimize_inference",
            description="Suggest optimizations for model inference",
            arguments=[
                PromptArgument(
                    name="model_name",
                    description="Name of the model to optimize",
                    required=True
                )
            ]
        )
    ]


@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
    """Get a prompt template with arguments filled in."""

    if name == "debug_deployment":
        ns = arguments.get("namespace", "default")
        deploy = arguments.get("deployment_name", "unknown")
        return GetPromptResult(
            description=f"Debug deployment {deploy} in {ns}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""I need help debugging a failing deployment in OpenShift.

Namespace: {ns}
Deployment: {deploy}

Please:
1. Use the get_pods tool to check pod status in the namespace
2. Identify any pods that are not Ready or have restarts
3. Suggest potential causes and remediation steps

Focus on common issues like:
- Image pull errors
- Resource constraints
- Readiness/liveness probe failures
- ConfigMap/Secret mounting issues"""
                    )
                )
            ]
        )

    elif name == "optimize_inference":
        model = arguments.get("model_name", "unknown")
        return GetPromptResult(
            description=f"Optimize inference for {model}",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Help me optimize the inference performance for model: {model}

Please analyze:
1. Current model status and configuration
2. Replica count and scaling settings
3. GPU utilization if applicable

Suggest optimizations for:
- Batch size tuning
- Model quantization options
- Horizontal scaling strategies
- Caching and request routing"""
                    )
                )
            ]
        )

    raise ValueError(f"Unknown prompt: {name}")


# ============================================================================
# Main entry point
# ============================================================================

async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```


## Step 3: Creating the Container Image {#step-3-creating-the-container-image}

Create a `Containerfile` for deployment:

```dockerfile
FROM registry.access.redhat.com/ubi9/python-311:latest

WORKDIR /app

# Copy requirements and install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir "mcp[cli]" httpx kubernetes openshift-client

# Copy application code
COPY server.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the MCP server
CMD ["python", "server.py"]
```

Build and push the image:

```bash
# Build the container image
podman build -t quay.io/yourusername/openshift-mcp-server:latest .

# Push to registry
podman push quay.io/yourusername/openshift-mcp-server:latest
```


## Step 4: Deploying to OpenShift {#step-4-deploying-to-openshift}

Create the Kubernetes manifests. First, `deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
  namespace: mcp-server-demo
  labels:
    app: mcp-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      serviceAccountName: mcp-server-sa
      containers:
        - name: mcp-server
          image: quay.io/yourusername/openshift-mcp-server:latest
          ports:
            - containerPort: 8080
              name: http
          env:
            - name: OPENSHIFT_API_URL
              value: "https://kubernetes.default.svc"
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
```

Create a `service-account.yaml` with appropriate permissions:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mcp-server-sa
  namespace: mcp-server-demo
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: mcp-server-reader
rules:
  - apiGroups: [""]
    resources: ["pods", "namespaces", "services", "configmaps"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments", "replicasets"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["serving.kserve.io"]
    resources: ["inferenceservices"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: mcp-server-reader-binding
subjects:
  - kind: ServiceAccount
    name: mcp-server-sa
    namespace: mcp-server-demo
roleRef:
  kind: ClusterRole
  name: mcp-server-reader
  apiGroup: rbac.authorization.k8s.io
```

And expose it with a `service.yaml`:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mcp-server
  namespace: mcp-server-demo
spec:
  selector:
    app: mcp-server
  ports:
    - port: 8080
      targetPort: 8080
      name: http
---
apiVersion: route.openshift.io/v1
kind: Route
metadata:
  name: mcp-server
  namespace: mcp-server-demo
spec:
  to:
    kind: Service
    name: mcp-server
  port:
    targetPort: http
  tls:
    termination: edge
```

Apply the manifests:

```bash
oc apply -f service-account.yaml
oc apply -f deployment.yaml
oc apply -f service.yaml

# Verify deployment
oc get pods -n mcp-server-demo
oc get route mcp-server -n mcp-server-demo
```

{{< figure src="/images/mcp-server-pod-running.png" caption="<span class=\"figure-number\">Figure 2: </span>MCP Server pod running in OpenShift" width="800px" >}}


## Step 5: Configuring Claude Desktop {#step-5-configuring-claude-desktop}

To use your MCP server with Claude Desktop, update your configuration at `~/.config/claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "openshift": {
      "command": "python",
      "args": ["/path/to/openshift-mcp-server/server.py"],
      "env": {
        "KUBECONFIG": "/path/to/.kube/config"
      }
    }
  }
}
```

For the remote deployment on OpenShift, you'll need to use the SSE transport. Modify `server.py` to support HTTP:

```python
# Add to server.py for HTTP transport support
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route

# Create SSE transport
sse = SseServerTransport("/messages")

async def handle_sse(request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0], streams[1],
            server.create_initialization_options()
        )

# Starlette app for HTTP serving
app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Route("/messages", endpoint=sse.handle_post_message, methods=["POST"]),
    ]
)
```


## Step 6: Testing the Integration {#step-6-testing-the-integration}

With everything deployed, let's test the integration. In Claude Desktop or your MCP client:

```text
User: What pods are running in the mcp-server-demo namespace?

Claude: I'll check the pods in that namespace using the OpenShift MCP server.

[Calls get_pods tool with namespace="mcp-server-demo"]

Here are the pods running in mcp-server-demo:

| Pod Name                    | Status  | Ready | Restarts |
|-----------------------------+---------+-------+----------|
| mcp-server-7d4f8b6c9-x2k4m | Running | True  |        0 |

The MCP server pod is running successfully with no restarts.
```


## Deploying a Model with OpenShift AI {#deploying-a-model-with-openshift-ai}

To complete the picture, let's deploy a model that our MCP server can interact with. Create `inferenceservice.yaml`:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: llama-7b
  namespace: mcp-server-demo
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    model:
      modelFormat:
        name: vLLM
      runtime: vllm-runtime
      storageUri: s3://models/llama-7b-hf
    resources:
      limits:
        nvidia.com/gpu: 1
      requests:
        memory: 16Gi
        cpu: 4
```

{{< figure src="/images/model-deployment-status.png" caption="<span class=\"figure-number\">Figure 3: </span>Model deployment status in OpenShift AI dashboard" width="800px" >}}


## Security Considerations {#security-considerations}

When deploying MCP servers in production, consider:

1.  **Authentication**: Implement OAuth2 or mTLS for server access
2.  **Network Policies**: Restrict which pods can communicate with the MCP server
3.  **RBAC Scoping**: Use namespace-scoped roles instead of cluster roles when possible
4.  **Secrets Management**: Use OpenShift secrets or external vault for credentials
5.  **Audit Logging**: Enable Kubernetes audit logging for compliance

<!--listend-->

```yaml
# Example NetworkPolicy to restrict access
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: mcp-server-policy
  namespace: mcp-server-demo
spec:
  podSelector:
    matchLabels:
      app: mcp-server
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              mcp-access: "true"
      ports:
        - port: 8080
```


## Extending the Server {#extending-the-server}

The beauty of MCP is its extensibility. Here are ideas for additional capabilities:

-   **Log Analysis**: Add a tool to fetch and analyze pod logs
-   **Metrics Integration**: Query Prometheus for cluster metrics
-   **Pipeline Triggers**: Start Tekton pipelines from the LLM
-   **Alert Responses**: Read and acknowledge alerts from AlertManager
-   **Documentation Search**: Index and search your runbooks


## Conclusion {#conclusion}

We've built a functional MCP server that bridges Claude's capabilities with OpenShift AI infrastructure. This pattern enables powerful workflows:

-   Natural language queries for cluster status
-   AI-assisted debugging of deployments
-   Automated inference requests to deployed models
-   Standardized tool interfaces for enterprise AI

The Model Context Protocol is still evolving, and the ecosystem of tools and integrations is growing rapidly. By combining MCP with OpenShift AI's robust model serving capabilities, you're well-positioned to build production-ready AI applications.


## Resources {#resources}

-   [Model Context Protocol Documentation](https://modelcontextprotocol.io/)
-   [OpenShift AI Documentation](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_self-managed/)
-   [Anthropic MCP Cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/misc/mcp)
-   [KServe Documentation](https://kserve.github.io/website/)
-   [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)

_The complete source code for this project is available on [GitHub](https://github.com/pebcac/openshift-mcp-server)._
