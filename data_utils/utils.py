import torch

def remove_undirected_duplicates(edges: torch.Tensor) -> torch.Tensor:
    """
    Removes duplicates in an undirected sense:
      - If edges has (u, v) and (v, u), we keep only one instance.
      - Also removes exact duplicates (u, v) repeated multiple times.
    
    Args:
        edges: (E, 2) tensor of edge indices.

    Returns:
        edges_uniq: (E_unique, 2) tensor of edges with duplicates removed,
                    always in sorted order [min(u,v), max(u,v)].
    """
    # 1) Sort each pair so edges[i] = [min(u,v), max(u,v)]
    edges_sorted, _ = edges.sort(dim=1)  # shape: (E, 2)

    # 2) Remove duplicates along the row dimension
    edges_uniq = torch.unique(edges_sorted, dim=0)

    return edges_uniq