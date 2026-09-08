"""
generate_large_scale_dataset.py
Generates a large-scale (1,000 workflows) realistic desktop workflow memory benchmark
spanning 10 diverse technical domains, with execution DAGs, checkpoints, timestamps,
application states, and diverse natural language continuation queries.
"""

import os
import sys
import json
import random
import time
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED)

DOMAINS = [
    "Python development",
    "Java development",
    "Web development",
    "Debugging",
    "Browser research",
    "Document editing",
    "File management",
    "Data analysis",
    "Presentation creation",
    "Software installation"
]

DOMAIN_CONFIGS = {
    "Python development": {
        "apps": ["Visual Studio Code", "Terminal", "PyCharm"],
        "files": ["main.py", "models.py", "requirements.txt", "test_core.py", "config.py", "app.py", "utils.py", "Dockerfile"],
        "browser_tabs": ["FastAPI Documentation", "PyTorch Docs", "Stack Overflow - Python", "GitHub Repository"],
        "templates": [
            ("Build a {framework} REST API for {service}", "Create and deploy a high-performance {framework} API handling {service} requests with validation and automated tests.",
             ["FastAPI", "Flask", "Django Ninja", "Tornado"], ["payment processing", "user authentication", "telemetry ingestion", "inventory sync", "notification dispatch", "document parsing", "metrics collection"]),
            ("Implement {tool} background task pipeline for {service}", "Configure asynchronous job worker queue using {tool} with Redis broker to process {service}.",
             ["Celery", "RQ", "Dramatiq", "Huey"], ["data preprocessing", "email newsletter delivery", "report generation", "video transcoding", "model batch inference"]),
            ("Create {lib} web scraper for {target}", "Extract structured tabular data from {target} using {lib} with rate limiting, proxy rotation, and SQLite persistence.",
             ["BeautifulSoup", "Playwright", "Scrapy", "Selenium"], ["e-commerce pricing", "real estate listings", "news articles", "job postings", "academic papers"]),
            ("Train {framework} neural network for {task}", "Implement, train, and evaluate a deep learning {framework} model from scratch for {task} with tensorboard logging.",
             ["PyTorch", "PyTorch Lightning", "Keras"], ["image classification", "text sentiment classification", "time series forecasting", "anomaly detection"])
        ],
        "node_actions": [
            "Create project directory structure and git repository",
            "Set up virtual environment and install requirements",
            "Implement core business logic and models",
            "Write unit tests with pytest and mock dependencies",
            "Configure linting with flake8 and black code formatter",
            "Build Docker container and verify containerized run",
            "Run integration tests against test database",
            "Deploy application to staging cluster"
        ]
    },
    "Java development": {
        "apps": ["IntelliJ IDEA", "Terminal", "Docker Desktop"],
        "files": ["Application.java", "pom.xml", "build.gradle", "application.properties", "UserController.java", "UserService.java", "DatabaseConfig.java"],
        "browser_tabs": ["Spring Boot Documentation", "Baeldung Spring Guides", "Maven Central Repository"],
        "templates": [
            ("Develop Spring Boot microservice for {service}", "Implement enterprise Spring Boot application with Spring Data JPA and REST controllers for {service}.",
             ["Spring Boot", "Micronaut", "Quarkus"], ["order management", "billing ledger", "user directory", "product catalog", "shipping tracking"]),
            ("Implement {tool} event streaming for {domain_task}", "Configure message producer, consumer group, and schema registry using {tool} for real-time {domain_task}.",
             ["Apache Kafka", "RabbitMQ", "ActiveMQ Artemis"], ["financial transactions", "sensor telemetry", "audit logging", "customer clickstream"]),
            ("Configure Gradle multi-module project for {system}", "Set up root build script, dependency management, and subproject configurations for {system}.",
             ["enterprise core", "banking backend", "e-commerce engine", "telecom platform"], ["build pipeline", "unit testing", "code coverage"]),
            ("Build Maven desktop client with {ui_framework}", "Develop cross-platform Java desktop application with {ui_framework} and SQLite persistence.",
             ["JavaFX", "Swing"], ["warehouse scanner", "pos terminal", "network monitor", "log viewer"])
        ],
        "node_actions": [
            "Initialize Gradle project with wrapper and dependencies",
            "Configure application.properties and datasource connection",
            "Implement domain entities and Spring Data repositories",
            "Develop REST controller endpoints and DTO mappers",
            "Write JUnit 5 tests and Mockito service verifications",
            "Configure Flyway database migration scripts",
            "Run Maven/Gradle build and assemble JAR package",
            "Execute end-to-end integration test suite"
        ]
    },
    "Web development": {
        "apps": ["Visual Studio Code", "Google Chrome", "Terminal"],
        "files": ["App.tsx", "package.json", "tailwind.config.js", "Navbar.tsx", "page.tsx", "global.css", "api.ts"],
        "browser_tabs": ["Next.js Documentation", "Tailwind CSS Components", "React DevTools", "Figma Design Mockups"],
        "templates": [
            ("Build responsive {framework} dashboard with {style}", "Create interactive admin portal using {framework} with {style}, theme switching, and data tables.",
             ["React", "Next.js", "Vue 3", "SvelteKit"], ["Tailwind CSS", "Material UI", "Chakra UI", "Styled Components"]),
            ("Implement full-stack {stack} application for {app_type}", "Set up client frontend, server API endpoints, and database connection for {app_type} using {stack}.",
             ["Next.js and Prisma", "Nuxt and Supabase", "MERN stack", "Remix and SQLite"], ["kanban task board", "collaborative whiteboard", "markdown blog", "crm system"]),
            ("Configure {bundler} build optimization and asset pipeline", "Set up code splitting, tree shaking, asset compression, and bundle analysis with {bundler}.",
             ["Vite", "Webpack 5", "Turbopack", "Rollup"], ["production build", "staging environment", "microfrontend host"]),
            ("Develop accessible design system components in {lang}", "Build reusable WCAG-compliant UI component library with Storybook and unit tests in {lang}.",
             ["TypeScript", "React TypeScript", "Web Components"], ["design tokens", "button primitives", "modal dialogs", "data pickers"])
        ],
        "node_actions": [
            "Scaffold frontend project with template and package manager",
            "Configure styling framework, typography, and color tokens",
            "Implement reusable UI layout and navigation components",
            "Integrate client-side state management and store",
            "Connect API client with React Query / SWR hooks",
            "Write Cypress / Playwright end-to-end user tests",
            "Optimize bundle size and purge unused CSS styles",
            "Deploy production build to CDN hosting provider"
        ]
    },
    "Debugging": {
        "apps": ["Visual Studio Code", "Terminal", "Google Chrome", "Wireshark"],
        "files": ["error.log", "crash_dump.dmp", "server.js", "nginx.conf", "main.py", "docker-compose.yml"],
        "browser_tabs": ["Datadog APM Dashboard", "Sentry Issue Tracker", "Stack Overflow", "GitHub Issues"],
        "templates": [
            ("Debug {issue_type} in {target_env}", "Profile execution, examine memory dumps, and isolate root cause of {issue_type} in {target_env}.",
             ["memory leak", "CPU spike", "thread deadlock", "socket leak", "heap fragmentation"], ["production worker pool", "Node.js cluster", "JVM container", "Go background daemon"]),
            ("Troubleshoot {status_code} errors on {server}", "Analyze access logs, upstream timeout configs, and SSL handshake errors causing {status_code} on {server}.",
             ["502 Bad Gateway", "504 Gateway Timeout", "500 Internal Server Error", "403 Forbidden"], ["Nginx reverse proxy", "HAProxy load balancer", "Traefik ingress", "Cloudflare edge"]),
            ("Fix race condition in {component}", "Reproduce intermittent race conditions using stress tests, add atomic locks, and verify thread safety in {component}.",
             ["distributed lock manager", "cache invalidation hook", "session store", "shopping cart checkout"], ["high-throughput pipeline", "concurrent worker queue"]),
            ("Investigate database query timeout on {db}", "Profile slow SQL queries, examine EXPLAIN query plans, and add missing composite indexes on {db}.",
             ["PostgreSQL 15", "MySQL 8", "MongoDB replica set", "Redis cluster"], ["analytics queries", "user transactions", "batch reports"])
        ],
        "node_actions": [
            "Reproduce error condition with deterministic test case",
            "Inspect stack traces and capture heap memory snapshot",
            "Attach debugger and step through suspect execution path",
            "Identify root-cause concurrency bug or memory leak",
            "Implement defensive patch and mutex synchronization",
            "Run regression test suite under sustained high load",
            "Validate fix in isolated staging environment",
            "Document root cause and preventive guidelines in postmortem"
        ]
    },
    "Browser research": {
        "apps": ["Google Chrome", "Obsidian", "Notion"],
        "files": ["research_notes.md", "bibliography.bib", "sources.csv", "summary.docx"],
        "browser_tabs": ["Google Scholar", "arXiv Computer Science", "ACM Digital Library", "GitHub Trending", "TechCrunch"],
        "templates": [
            ("Research competitor pricing and features for {domain}", "Collect, summarize, and synthesize market pricing tiers, trial models, and feature matrices across top {domain} competitors.",
             ["cloud hosting", "developer tools", "email marketing", "AI coding assistants", "project management tools"], ["market overview", "feature comparison"]),
            ("Investigate security CVE disclosures for {lib}", "Review National Vulnerability Database, GitHub security advisories, and patch notes for {lib}.",
             ["OpenSSL", "Log4j", "Node fetch", "Django", "Spring Framework", "urllib3"], ["critical severity", "remote code execution"]),
            ("Conduct literature review on {research_topic}", "Search Google Scholar, arXiv, and ACM Digital Library for recent papers on {research_topic} and synthesize key findings.",
             ["retrieval-augmented generation", "agentic memory architectures", "graph neural networks", "diffusion models for planning"], ["survey paper", "state of the art"]),
            ("Synthesize user feedback and pain points for {product}", "Analyze community forum threads, GitHub discussions, and reviews regarding {product} usability and reliability.",
             ["VS Code extensions", "Docker Desktop", "Kubernetes k9s", "JupyterLab 4"], ["developer sentiment", "bug reports"])
        ],
        "node_actions": [
            "Search academic databases and tech documentation for query",
            "Filter and bookmark high-authority primary sources",
            "Extract key qualitative insights and performance figures",
            "Cross-reference conflicting claims across multiple reports",
            "Synthesize structured findings into comparison table",
            "Draft executive summary with actionable recommendations",
            "Compile comprehensive bibliography with verified URLs"
        ]
    },
    "Document editing": {
        "apps": ["Microsoft Word", "LibreOffice Writer", "Overleaf (Chrome)"],
        "files": ["design_doc.docx", "architecture_rfc.md", "manuscript.tex", "references.bib", "draft_v2.pdf"],
        "browser_tabs": ["Overleaf Project", "Grammarly Editor", "Company Confluence Wiki", "Google Drive"],
        "templates": [
            ("Draft {doc_type} for {project}", "Author comprehensive and structured {doc_type} covering architecture, endpoints, security, and setup for {project}.",
             ["technical design document", "system architecture RFC", "security compliance whitepaper", "disaster recovery plan"], ["cloud migration", "v2 auth rewrite", "data pipeline", "microservices"]),
            ("Format academic conference manuscript in LaTeX for {venue}", "Typeset paper sections, format figures, bibliography citations, and align IEEE/ACM template for {venue}.",
             ["NeurIPS", "ICLR", "ACM SIGMOD", "IEEE S&P", "VLDB"], ["camera ready submission", "initial review blind draft"]),
            ("Write comprehensive API reference documentation in {format}", "Document request schemas, response codes, error payloads, and code samples in {format}.",
             ["OpenAPI 3.1 Swagger", "Markdown Mintlify", "Redoc", "Docusaurus"], ["public developer portal", "partner integration guide"]),
            ("Compile quarterly engineering retrospective and metrics report", "Aggregate velocity charts, incident postmortems, SLO compliance, and OKR progress into executive report in {tool}.",
             ["Microsoft Word", "Google Docs", "Notion", "Confluence"], ["Q1 engineering retrospective", "annual infrastructure report"])
        ],
        "node_actions": [
            "Outline document structure, sections, and target audience",
            "Draft technical content for introductory and core chapters",
            "Create high-resolution architecture diagrams and figures",
            "Format document styles, headings, tables, and typography",
            "Perform comprehensive technical review and copy editing",
            "Generate final PDF export and verify visual fidelity",
            "Distribute draft to stakeholders for collaborative review"
        ]
    },
    "File management": {
        "apps": ["Terminal", "File Explorer", "7-Zip", "Cyberduck"],
        "files": ["archive_2026.tar.gz", "sync_manifest.json", "file_inventory.csv", "backup.sh"],
        "browser_tabs": ["AWS S3 Management Console", "Google Cloud Storage", "Backblaze B2 Console"],
        "templates": [
            ("Organize and catalog {media_type} files by date and project", "Traverse directory trees, parse metadata tags, create directory hierarchy, and move {media_type} into structured archives.",
             ["raw camera footage", "design asset PSDs", "historical server logs", "dataset parquet partitions", "scan PDFs"], ["quarterly backup", "long-term archive"]),
            ("Automate daily compressed backup and upload to {storage}", "Write automated backup script with tar compression, AES-256 encryption, checksum verification, and sync to {storage}.",
             ["AWS S3 glacier", "Google Cloud Storage", "Backblaze B2", "remote SFTP server"], ["database dumps", "media uploads"]),
            ("Purge stale cache and rotate system log files", "Audit disk storage consumption, find files older than 30 days, clean package manager caches, and configure logrotate for {daemon}.",
             ["Docker build cache", "systemd journal", "Nginx access logs", "pip and conda cache"], ["cleanup maintenance", "disk optimization"]),
            ("Batch rename and standardize naming convention for {file_type}", "Parse regex tokens, normalize dates and prefixes, and rename thousands of {file_type} without collisions.",
             ["telemetry CSV exports", "billing invoice PDFs", "product image assets", "audio WAV recordings"], ["standardized format", "sanitized directory"])
        ],
        "node_actions": [
            "Scan directory hierarchy and inventory file types and sizes",
            "Identify duplicate files and zero-byte corrupted items",
            "Design canonical directory categorization schema",
            "Write automated script to batch rename and move files",
            "Compute SHA-256 checksums to guarantee data integrity",
            "Compress archived directories into encrypted archives",
            "Sync finalized archives to remote offsite cold storage"
        ]
    },
    "Data analysis": {
        "apps": ["JupyterLab", "Visual Studio Code", "Google Chrome"],
        "files": ["analysis.ipynb", "dataset.parquet", "clean_data.csv", "retention_chart.png", "eda_report.html"],
        "browser_tabs": ["JupyterLab Workspace", "Plotly Documentation", "Pandas API Reference", "Kaggle Datasets"],
        "templates": [
            ("Analyze customer churn and cohort retention in {tool}", "Load transaction records, compute monthly retention cohorts, calculate survival probabilities, and plot retention heatmap in {tool}.",
             ["Pandas and Seaborn", "Polars and Matplotlib", "DuckDB and Plotly", "R Tidyverse"], ["cohort retention", "churn modeling"]),
            ("Perform exploratory spatial data analysis on {dataset}", "Merge shapefiles with demographic statistics, calculate Moran's I spatial autocorrelation, and render choropleth map for {dataset}.",
             ["census tracts", "wildfire risk zones", "urban transit ridership", "retail foot traffic"], ["spatial cluster analysis", "density mapping"]),
            ("Build interactive financial metrics dashboard with {lib}", "Aggregate revenue streams, calculate run rates, visualize burn rates, and display interactive charts using {lib}.",
             ["Streamlit", "Dash", "Panel", "Gradio"], ["SaaS revenue dashboard", "real-time portfolio tracker"]),
            ("Run hypothesis testing and A/B test analysis for {metric}", "Check statistical normality, compute two-sample t-test, estimate p-values and confidence intervals for {metric}.",
             ["conversion rate lift", "user session duration", "checkout drop-off", "onboarding completion"], ["product experiment", "growth optimization"])
        ],
        "node_actions": [
            "Ingest raw CSV/Parquet datasets into dataframe",
            "Clean missing values, cast datatypes, and filter outliers",
            "Compute summary statistics, correlation matrices, and metrics",
            "Generate exploratory distribution plots and scatter matrices",
            "Perform statistical significance testing and hypothesis checks",
            "Build predictive regression / classification baseline",
            "Export final analytical report and interactive visual charts"
        ]
    },
    "Presentation creation": {
        "apps": ["Microsoft PowerPoint", "Google Slides (Chrome)", "Keynote"],
        "files": ["pitch_deck_v4.pptx", "q3_qbr_slides.pptx", "system_architecture.svg", "financial_charts.xlsx"],
        "browser_tabs": ["Google Slides Editor", "Pitch Deck Examples", "Unsplash Free Stock Photos", "Canva Graphics"],
        "templates": [
            ("Design investor pitch deck for {startup_theme}", "Create compelling visual narrative, problem-solution slides, market sizing TAM/SAM, unit economics, and roadmap for {startup_theme}.",
             ["AI developer tool", "climate fintech platform", "enterprise cybersecurity", "autonomous robotics startup"], ["seed round pitch", "series A presentation"]),
            ("Build technical architecture slides for {meeting}", "Diagram system components, sequence diagrams, network topologies, and data flow for {meeting}.",
             ["engineering all-hands", "architecture review board", "security council", "client technical kickoff"], ["cloud migration roadmap", "microservices transition"]),
            ("Create executive quarterly business review presentation", "Format revenue trends, customer logos, product delivery milestones, and headcount projections in {app}.",
             ["Google Slides", "Keynote", "PowerPoint", "Marp Markdown"], ["quarterly business review", "board meeting presentation"]),
            ("Prepare workshop presentation slides on {topic}", "Structure learning objectives, hands-on lab instructions, interactive quiz prompts, and cheat sheets on {topic}.",
             ["Docker containerization", "Kubernetes fundamentals", "Git advanced branching", "Prompt engineering"], ["engineering bootcamp", "developer training"])
        ],
        "node_actions": [
            "Define presentation objectives, key takeaways, and narrative arc",
            "Draft slide-by-slide storyboard and talking points",
            "Design visual theme, master slides, and brand typography",
            "Create data charts, process flows, and architectural diagrams",
            "Refine slide copy for maximum clarity and visual impact",
            "Rehearse timing and add speaker notes for each slide",
            "Export slide deck to PDF and presentation formats"
        ]
    },
    "Software installation": {
        "apps": ["Terminal", "PowerShell", "Docker Desktop"],
        "files": ["install.sh", "docker-compose.yml", "daemon.json", "env_vars.sh", "install_log.txt"],
        "browser_tabs": ["Docker Hub", "Kubernetes Official Docs", "GitHub Releases", "Ubuntu Packages"],
        "templates": [
            ("Install and configure {software} on {os_platform}", "Verify prerequisite packages, download official release binaries, configure environment variables, and initialize system daemon for {software} on {os_platform}.",
             ["PostgreSQL 16", "Redis 7", "Docker Desktop with WSL2", "Elasticsearch 8", "Nginx", "RabbitMQ"], ["Ubuntu 22.04 LTS", "Debian 12", "Fedora 39", "Arch Linux", "Windows 11 WSL"]),
            ("Set up {toolkit} deep learning environment", "Install NVIDIA drivers, configure CUDA 12.2 and cuDNN, create conda environment, and verify GPU acceleration in {toolkit}.",
             ["PyTorch and TorchVision", "TensorFlow GPU", "JAX and Flax", "ONNX Runtime GPU"], ["machine learning workstation", "GPU cloud instance"]),
            ("Install and configure {dev_tool} CLI with multi-account credentials", "Download binary, verify GPG signature, configure authentication profiles, and test API connectivity for {dev_tool}.",
             ["AWS CLI v2", "Google Cloud SDK", "Terraform and Tofu", "Kubernetes kubectl and helm"], ["cloud engineer workstation", "CI/CD runner machine"]),
            ("Configure local development cluster with {cluster_tool}", "Initialize single-node Kubernetes cluster, deploy ingress controller, set up metrics server, and install dashboard using {cluster_tool}.",
             ["Minikube", "Kind", "K3s", "MicroK8s"], ["local microservices testing", "developer sandbox"])
        ],
        "node_actions": [
            "Check system hardware requirements and OS prerequisites",
            "Download verified package installer or binary release",
            "Verify package cryptographic GPG signature and SHA-256 checksum",
            "Run automated installer and accept software licensing",
            "Configure configuration file and set environment variables",
            "Start system service daemon and enable launch on boot",
            "Run automated smoke tests to verify healthy installation"
        ]
    }
}

def generate_large_scale_workflows(num_workflows=1000, output_path="large_scale_workflows.json"):
    print(f"Generating {num_workflows} structured workflows across {len(DOMAINS)} domains...")
    now = datetime(2026, 9, 6, 12, 0, 0)
    workflows = []
    
    per_domain = num_workflows // len(DOMAINS)
    extra = num_workflows % len(DOMAINS)
    
    wf_counter = 1
    for d_idx, domain in enumerate(DOMAINS):
        cfg = DOMAIN_CONFIGS[domain]
        count = per_domain + (1 if d_idx < extra else 0)
        
        for i in range(count):
            wf_id = f"workflow_{wf_counter:03d}"
            wf_counter += 1
            
            tmpl = cfg["templates"][i % len(cfg["templates"])]
            goal_fmt, desc_fmt, list1, list2 = tmpl
            v1 = random.choice(list1)
            v2 = random.choice(list2)
            
            # Format goal and description
            goal = goal_fmt.replace("{framework}", v1).replace("{tool}", v1).replace("{lib}", v1).replace("{stack}", v1).replace("{bundler}", v1).replace("{issue_type}", v1).replace("{status_code}", v1).replace("{doc_type}", v1).replace("{media_type}", v1).replace("{startup_theme}", v1).replace("{software}", v1).replace("{toolkit}", v1).replace("{dev_tool}", v1).replace("{cluster_tool}", v1).replace("{service}", v2).replace("{target}", v2).replace("{task}", v2).replace("{domain_task}", v2).replace("{system}", v2).replace("{ui_framework}", v2).replace("{style}", v2).replace("{app_type}", v2).replace("{lang}", v2).replace("{target_env}", v2).replace("{server}", v2).replace("{component}", v2).replace("{db}", v2).replace("{domain}", v2).replace("{research_topic}", v2).replace("{product}", v2).replace("{project}", v2).replace("{venue}", v2).replace("{format}", v2).replace("{storage}", v2).replace("{daemon}", v2).replace("{file_type}", v2).replace("{dataset}", v2).replace("{metric}", v2).replace("{meeting}", v2).replace("{app}", v2).replace("{topic}", v2).replace("{os_platform}", v2)
            desc = desc_fmt.replace("{framework}", v1).replace("{tool}", v1).replace("{lib}", v1).replace("{stack}", v1).replace("{bundler}", v1).replace("{issue_type}", v1).replace("{status_code}", v1).replace("{doc_type}", v1).replace("{media_type}", v1).replace("{startup_theme}", v1).replace("{software}", v1).replace("{toolkit}", v1).replace("{dev_tool}", v1).replace("{cluster_tool}", v1).replace("{service}", v2).replace("{target}", v2).replace("{task}", v2).replace("{domain_task}", v2).replace("{system}", v2).replace("{ui_framework}", v2).replace("{style}", v2).replace("{app_type}", v2).replace("{lang}", v2).replace("{target_env}", v2).replace("{server}", v2).replace("{component}", v2).replace("{db}", v2).replace("{domain}", v2).replace("{research_topic}", v2).replace("{product}", v2).replace("{project}", v2).replace("{venue}", v2).replace("{format}", v2).replace("{storage}", v2).replace("{daemon}", v2).replace("{file_type}", v2).replace("{dataset}", v2).replace("{metric}", v2).replace("{meeting}", v2).replace("{app}", v2).replace("{topic}", v2).replace("{os_platform}", v2)
            
            # Nodes: 4 to 8 nodes
            node_actions = cfg["node_actions"]
            num_nodes = random.randint(4, min(8, len(node_actions)))
            selected_actions = node_actions[:num_nodes]
            
            # Status distribution: FAILED (30%), INCOMPLETE (45%), COMPLETED (25%)
            overall_status = random.choices(["FAILED", "INCOMPLETE", "COMPLETED"], weights=[0.30, 0.45, 0.25])[0]
            
            nodes = []
            if overall_status == "COMPLETED":
                for n_idx, act in enumerate(selected_actions):
                    nodes.append({
                        "node_id": n_idx + 1,
                        "description": act,
                        "status": "COMPLETED"
                    })
                active_node_id = num_nodes
                state_summary = "All steps completed successfully. Final artifacts generated."
            elif overall_status == "INCOMPLETE":
                split_k = random.randint(1, num_nodes - 1)
                for n_idx, act in enumerate(selected_actions):
                    st = "COMPLETED" if n_idx < split_k else "INCOMPLETE"
                    nodes.append({
                        "node_id": n_idx + 1,
                        "description": act,
                        "status": st
                    })
                active_node_id = split_k
                state_summary = f"Workflow paused after step {split_k}. Next step ready for execution."
            else: # FAILED
                split_k = random.randint(1, num_nodes - 1)
                for n_idx, act in enumerate(selected_actions):
                    if n_idx < split_k:
                        st = "COMPLETED"
                    elif n_idx == split_k:
                        st = "FAILED"
                    else:
                        st = "INCOMPLETE"
                    nodes.append({
                        "node_id": n_idx + 1,
                        "description": act,
                        "status": st
                    })
                active_node_id = split_k + 1
                state_summary = f"Execution failed at step {active_node_id} with runtime exception. Intervention required."
                
            # DAG dependencies (sequential with occasional skip edges)
            dependencies = []
            for n_idx in range(1, len(nodes)):
                dependencies.append([n_idx, n_idx + 1])
                if n_idx >= 2 and random.random() < 0.25:
                    dependencies.append([n_idx - 1, n_idx + 1])
                    
            # Timestamp (exponentially distributed over past 30 days)
            days_ago = random.expovariate(0.15)
            days_ago = min(days_ago, 30.0)
            checkpoint_time = (now - timedelta(days=days_ago)).isoformat()
            
            # Application state context matching domain
            selected_apps = random.sample(cfg["apps"], k=min(2, len(cfg["apps"])))
            selected_files = random.sample(cfg["files"], k=min(3, len(cfg["files"])))
            selected_tabs = random.sample(cfg["browser_tabs"], k=min(2, len(cfg["browser_tabs"])))
            
            current_state = {
                "applications": selected_apps,
                "files": selected_files,
                "browser_tabs": selected_tabs
            }
            
            # Realistic query formulations
            if overall_status == "FAILED":
                q_list = [
                    f"Fix and resume the failed {goal.lower()}",
                    f"Continue my work on {goal.lower()} that had an error",
                    f"Debug the issue with {goal.lower()}",
                    f"Recover the interrupted {domain.lower()} task",
                    f"Continue my previous work: {goal.lower()}"
                ]
            elif overall_status == "INCOMPLETE":
                q_list = [
                    f"Continue my {goal.lower()}",
                    f"Resume my previous work on {goal.lower()}",
                    f"Pick up where I left off on {goal.lower()}",
                    f"Finish the remaining steps for {goal.lower()}",
                    f"Continue the interrupted {domain.lower()} task"
                ]
            else:
                q_list = [
                    f"Review my previous completed work on {goal.lower()}",
                    f"Find the workflow where I did {goal.lower()}",
                    f"Retrieve the completed {domain.lower()} project for {goal.lower()}",
                    f"Show the workflow for {goal.lower()}",
                    f"Inspect the steps I took to {goal.lower()}"
                ]
                
            workflow_entry = {
                "workflow_id": wf_id,
                "domain": domain,
                "goal": goal,
                "description": desc,
                "overall_status": overall_status,
                "timestamp": checkpoint_time,
                "days_ago": round(days_ago, 2),
                "graph": {
                    "nodes": nodes,
                    "dependencies": dependencies
                },
                "checkpoint": {
                    "timestamp": checkpoint_time,
                    "last_active_node": active_node_id,
                    "state_summary": state_summary
                },
                "current_state": current_state,
                "queries": q_list
            }
            workflows.append(workflow_entry)
            
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(workflows, f, indent=2)
        
    print(f"Saved {len(workflows)} workflows to {output_path} ({os.path.getsize(output_path) / (1024*1024):.2f} MB).")
    total_q = sum(len(w["queries"]) for w in workflows)
    print(f"Total user queries generated: {total_q}")
    return workflows

if __name__ == "__main__":
    count = 1000
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
    generate_large_scale_workflows(count)
