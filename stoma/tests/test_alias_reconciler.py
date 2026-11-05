"""Tests for Ingest-Time Entity Alias Reconciler in Stoma (Item 116 / arXiv:2610.40482)."""

import pytest
from stoma.alias_reconciler import (
    CanonicalEntity,
    EntityAliasReconciler,
)


def test_alias_registration_and_resolution():
    """Verify alias registration maps surface forms and acronyms to canonical ID."""
    reconciler = EntityAliasReconciler()
    entity = reconciler.register_entity(
        canonical_id="ent:google_deepmind",
        primary_name="Google DeepMind",
        aliases=["DeepMind", "GDM", "Google Deep Mind"],
        entity_type="organization",
    )

    assert entity.canonical_id == "ent:google_deepmind"
    assert reconciler.resolve("Google DeepMind") == "ent:google_deepmind"
    assert reconciler.resolve("DeepMind") == "ent:google_deepmind"
    assert reconciler.resolve("GDM") == "ent:google_deepmind"
    assert reconciler.resolve("gdm") == "ent:google_deepmind"
    assert reconciler.resolve("Google Deep Mind") == "ent:google_deepmind"

    # Unknown entity resolves to itself
    assert reconciler.resolve("Unknown Entity 123") == "Unknown Entity 123"


def test_reconcile_graph_block_merging():
    """Verify node merging and edge rewiring across alias mentions."""
    reconciler = EntityAliasReconciler()
    reconciler.register_entity(
        canonical_id="ent:google_deepmind",
        primary_name="Google DeepMind",
        aliases=["DeepMind", "GDM"],
    )
    reconciler.register_entity(
        canonical_id="ent:alphafold",
        primary_name="AlphaFold 3",
        aliases=["AF3", "AlphaFold"],
    )

    nodes = [
        {"id": "GDM", "name": "GDM", "confidence": 0.95},
        {"id": "DeepMind", "name": "DeepMind", "location": "London"},
        {"id": "AF3", "name": "AF3", "version": 3},
    ]

    edges = [
        {"src": "DeepMind", "rel": "developed", "dst": "AF3"},
        # Self-loop alias collapse: GDM is DeepMind
        {"src": "GDM", "rel": "subsidiary_of", "dst": "DeepMind"},
    ]

    reconciled_nodes, reconciled_edges = reconciler.reconcile_graph_block(nodes, edges)

    # 3 raw nodes collapse into 2 canonical entities
    assert len(reconciled_nodes) == 2
    node_ids = {n["id"] for n in reconciled_nodes}
    assert node_ids == {"ent:google_deepmind", "ent:alphafold"}

    # Merged node metadata
    gdm_node = next(n for n in reconciled_nodes if n["id"] == "ent:google_deepmind")
    assert gdm_node["name"] == "Google DeepMind"
    assert gdm_node["confidence"] == 0.95
    assert gdm_node["location"] == "London"

    # Edges: self-loop dropped, valid edge rewired
    assert len(reconciled_edges) == 1
    edge = reconciled_edges[0]
    assert edge["src"] == "ent:google_deepmind"
    assert edge["dst"] == "ent:alphafold"
    assert edge["rel"] == "developed"
