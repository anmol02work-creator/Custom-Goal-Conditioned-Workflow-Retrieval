import json

def extract_continuation_subgraph(wf):
    nodes = wf.get('nodes', [])
    dependencies = wf.get('dependencies', [])
    
    comp_ids = set(n['node_id'] for n in nodes if n['status'] == 'COMPLETED')
    fail_ids = set(n['node_id'] for n in nodes if n['status'] == 'FAILED')
    inc_ids = set(n['node_id'] for n in nodes if n['status'] == 'INCOMPLETE')
    
    # Actionable nodes: incomplete or failed nodes whose upstream dependencies are met (or have no deps)
    upstream = {}
    for src, dst in dependencies:
        upstream.setdefault(dst, set()).add(src)
        
    actionable_node_ids = []
    for n in nodes:
        nid = n['node_id']
        if nid in fail_ids or nid in inc_ids:
            deps = upstream.get(nid, set())
            if deps.issubset(comp_ids) or nid in fail_ids:
                actionable_node_ids.append(nid)
                
    # Relevant node IDs for continuation: failed nodes + incomplete nodes
    relevant_node_ids = sorted(list(fail_ids | inc_ids))
    if not relevant_node_ids:
        relevant_node_ids = [n['node_id'] for n in nodes]
        
    subgraph_nodes = [n for n in nodes if n['node_id'] in relevant_node_ids]
    subgraph_deps = [[src, dst] for src, dst in dependencies if src in relevant_node_ids and dst in relevant_node_ids]
    
    return {
        "workflow_id": wf['workflow_id'],
        "goal": wf['goal'],
        "status": wf['status'],
        "checkpoint": wf['checkpoint'],
        "actionable_node_ids": actionable_node_ids,
        "retrieved_node_ids": relevant_node_ids,
        "subgraph_nodes": subgraph_nodes,
        "subgraph_dependencies": subgraph_deps
    }

from test_generator import generate_synthetic_workflows
wfs = generate_synthetic_workflows(10)
sample_wf = [w for w in wfs if w['status'] == 'FAILED'][0]
sub = extract_continuation_subgraph(sample_wf)
print("Continuation Subgraph for sample failed workflow:")
print(json.dumps(sub, indent=2))
