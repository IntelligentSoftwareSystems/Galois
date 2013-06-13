

std::vector<unsigned> edgeCut(GGraph& g, unsigned nparts) {
  std::vector<unsigned> cuts(nparts);

 //find boundary nodes with positive gain
  for (GNode n : g) {
    unsigned gPart = g.getData(n).getPart();
    for (auto ii = g.edge_begin(n), ee = g.edge_end(n); ii != ee; ++ii) {
      auto& m = g.getData(g.getEdgeDst(ii));
      if (m.getPart() != gPart) {
        cuts.at(gPart) += g.getEdgeData(ii);
      }
    }
  }

  return cuts;
}
