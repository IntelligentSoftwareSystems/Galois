
//Matches calls to getdat and binds getDataVar to the valiable expression which is the node
//g.getData(foo) // binds here, foo bound to getDataVar
MatchStatement findGetData = callExpr(callee(functionDecl(hasName("getData"))), hasArgument(0, expr().bind("getDataVar")));

//finds field indexing of references to graph nodes
//N& = g.getData(foo); // foo bound to getDataVar
//N.f; // binds here to fieldRef
MatchStatement findFieldOfNodeRef = memberExpr(has(declRefExpr(to(varDecl(hasInitializer(findGetData)))))).bind("fieldRef");

//finds references to fields of a graph node
//N& = g.getData(foo).f // foo bound to getDataVar, fieldRef bound here
//N // binds here
MatchStatement findRefOfFieldRef = declRefExpr(to(varDecl(hasInitializer(memberExpr(has(findGetData)).bind("fieldRef")))));

//finds direct uses of fields not assigned to references
// g.getData(foo).f 
MatchStatement findFieldUseDirect = memberExpr(has(findGetData), unless(hasAncestor(varDecl(hasType(referenceType()))))).bind("fieldRef");

