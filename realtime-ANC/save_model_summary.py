# --------------------------------------------------------------------------------------------------------------
# 
# Author - Vishal Shrivastava
# 
# ---------------------------------------------------------------------------------------------------------------
import tensorflow as tf
from anc_model import ANCModel
import networkx as nx
import matplotlib.pyplot as plt

def plot_model_architecture(model, file_path):
    """
    Plot the model architecture and save as an image.
    
    Args:
        model (tf.keras.Model): The Keras model instance.
        file_path (str): The path where the image will be saved.
    """
    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes and edges
    for layer in model.layers:
        graph.add_node(layer.name, label=layer.name)
        for node in layer.inbound_nodes:
            inbound_layers = node.inbound_layers if isinstance(node.inbound_layers, list) else [node.inbound_layers]
            for inbound_layer in inbound_layers:
                graph.add_edge(inbound_layer.name, layer.name)
    
    # Draw the graph
    pos = nx.spring_layout(graph)
    labels = nx.get_node_attributes(graph, 'label')
    nx.draw(graph, pos, labels=labels, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold', edge_color='gray')
    
    # Save the graph as an image
    plt.savefig(file_path)
    plt.show()
    print(f"Model architecture saved to {file_path}")

# Example usage
if __name__ == "__main__":
    # Define the input shape for the ANC model
    input_shape = (32, 1)  # Example input shape; modify as needed
    
    # Initialize the ANC model
    anc_model_instance = ANCModel(input_shape=input_shape)
    anc_model_instance.model.summary()

    # Plot the model architecture and save as an image
    # plot_model_architecture(anc_model_instance.model, "model_architecture.png")
