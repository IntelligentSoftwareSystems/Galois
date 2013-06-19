

std::vector<unsigned> edgeCut(GGraph& g, unsigned nparts) {
  std::vector<unsigned> cuts(nparts);

 //find boundary nodes with positive gain
  for (auto nn = g.begin(), en = g.end(); nn != en; ++nn) {
    unsigned gPart = g.getData(*nn).getPart();
    for (auto ii = g.edge_begin(*nn), ee = g.edge_end(*nn); ii != ee; ++ii) {
      auto& m = g.getData(g.getEdgeDst(ii));
      if (m.getPart() != gPart) {
        cuts.at(gPart) += g.getEdgeData(ii);
      }
    }
  }

  return cuts;
}
