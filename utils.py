import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import csv
import matplotlib.gridspec as gridspec
import seaborn as sns


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

def plot_solution(graph: dict, final_tour: list):
    G = nx.Graph()

    for key, value in graph.items():
        G.add_edge(key[0], key[1], weight=value)

    edges = [(u, v) for (u, v, d) in G.edges(data=True)]
    edges_solution = [(final_tour[i], final_tour[i + 1]) for i in range(len(final_tour) - 1)]


    pos = nx.spring_layout(G, seed=1)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)
    # edges
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.2)
    nx.draw_networkx_edges(G, pos, edgelist=edges_solution, width=2.5, alpha=0.9, edge_color="y")
    

    # node labels
    nx.draw_networkx_labels(G, pos, font_weight='normal', font_size=10, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_evolution(evolution):
    plt.plot(evolution)
    plt.show()


def build_instance(nodes, name, num = 20):
    #nodes = "abcdefghijklmnopqrstuvwxyz"
    chooser = list(range(1,num))
    name = "resources/" + name + ".csv"

    with open(name, 'w') as f:
        writer = csv.writer(f)
        for s in nodes:
            for t in nodes:
                if s != t:
                    writer.writerow([s,t,np.random.choice(chooser)])


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


def plot_results(graph: dict, final_tour: list, evolution, ants_evol):
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    G = nx.Graph()

    for key, value in graph.items():
        G.add_edge(key[0], key[1], weight=value)

    edges = [(u, v) for (u, v, d) in G.edges(data=True)]
    edges_solution = [(final_tour[i], final_tour[i + 1]) for i in range(len(final_tour) - 1)]
    pos = nx.spring_layout(G, seed=1)  # positions for all nodes - seed for reproducibility
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.2)
    nx.draw_networkx_edges(G, pos, edgelist=edges_solution, width=2.5, alpha=0.9, edge_color="y")
    nx.draw_networkx_labels(G, pos, font_weight='normal', font_size=10, font_family="sans-serif")
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()

    plt.subplot(1, 2, 2) # index 2
    plt.plot(evolution, color='b')
    plt.plot(ants_evol, color='r')
    plt.legend(["Best tour", "Average ant tour"])
    plt.show()

def plot_subset_solution(acs, evolution):
    plt.subplot(1, 2, 1) # row 1, col 2 index 1
    plt.bar(range(len(acs.pheromone)), acs.pheromone)

    plt.subplot(1, 2, 2) # index 2
    plt.plot(evolution, color='b')
    #plt.plot(ants_evol, color='r')
    #plt.legend(["Best tour", "Average ant tour"])
    plt.show()

def plot_subset_solution2(acs, evolution, ants_evol):
    gs = gridspec.GridSpec(2, 2)

    plt.figure()
    ax = plt.subplot(gs[0, 0]) # row 0, col 0
    plt.bar(range(len(acs.pheromone)), acs.pheromone)

    ax = plt.subplot(gs[0, 1]) # row 0, col 1
    plt.plot(evolution, color='b')
    plt.plot(ants_evol, color='r')

    ax = plt.subplot(gs[1, :]) # row 1, span all columns
    plt.plot(acs.pheromones)

    plt.show()

def plot_mkp_solution2(acs, evolution, ants_evol):
    gs = gridspec.GridSpec(2, 2)

    plt.figure()
    ax = plt.subplot(gs[0, 0]) # row 0, col 0
    heat_map = sns.heatmap(acs.pheromone, xticklabels=False, yticklabels=False, cmap="YlGnBu")
    plt.xlabel('Items')
    plt.ylabel('KPs')
    plt.title("Pheromone trails")
    #plt.bar(range(len(acs.pheromone.sum(axis=0))), acs.pheromone.sum(axis = 0))

    ax = plt.subplot(gs[0, 1]) # row 0, col 1
    heat_map = sns.heatmap(acs.best_fit, linewidths=.5, cmap='Blues')
    plt.xlabel('Items')
    plt.ylabel('KPs')
    plt.title("Solution")

    ax = plt.subplot(gs[1, :]) # row 1, span all columns
    plt.plot(evolution, color='b')
    plt.plot(ants_evol, color='r')
    plt.legend(["Best tour", "Average ant tour"])
    plt.title("Profit")
    plt.xlabel('iterations')
    plt.ylabel('Profit')

    plt.show()

def plot_mkp_solution3(acs, evolution):
    gs = gridspec.GridSpec(2, 2)

    plt.figure()
    ax = plt.subplot(gs[0, 0]) # row 0, col 0
    heat_map = sns.heatmap(acs.pheromone, xticklabels=False, yticklabels=False, cmap="YlGnBu")
    plt.xlabel('Items')
    plt.ylabel('KPs')
    plt.title("Pheromone trails")
    #plt.bar(range(len(acs.pheromone.sum(axis=0))), acs.pheromone.sum(axis = 0))

    ax = plt.subplot(gs[0, 1]) # row 0, col 1
    heat_map = sns.heatmap(acs.best_fit, linewidths=.5, cmap='Blues')
    plt.xlabel('Items')
    plt.ylabel('KPs')
    plt.title("Solution")

    ax = plt.subplot(gs[1, :]) # row 1, span all columns
    plt.plot(evolution, color='b')
    plt.legend(["Best tour"])
    plt.title("Profit")
    plt.xlabel('iterations')
    plt.ylabel('Profit')

    plt.show()