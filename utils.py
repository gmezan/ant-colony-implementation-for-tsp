import matplotlib.pyplot as plt
import networkx as nx
import csv

def plot_graph(graph: dict):
    G = nx.Graph()

    for key, value in graph.items():
        G.add_edge(key[0], key[1], weight=value)

    edges = [(u, v) for (u, v, d) in G.edges(data=True)]

    pos = nx.spring_layout(G, seed=1)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2)

    # node labels
    nx.draw_networkx_labels(G, pos, font_weight='normal', font_size=20, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# csv: source,destination,length
def read_instance(name: str) -> tuple:
    graph = {}
    nodes = []
    with open(name) as csvfile:
        routes = csv.reader(csvfile)
        for route in routes:
            # getting raw values
            src = route[0]
            dst = route[1]
            w = float(route[2])

            keys = list(graph.keys())
            
            if (((src,dst) not in keys) and ((dst,src) not in keys)):
                graph[(src, dst)] = w

            if src not in nodes:
                nodes.append(src)
                
            if dst not in nodes:
                nodes.append(dst)

    return graph, nodes