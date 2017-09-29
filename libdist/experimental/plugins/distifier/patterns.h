//Match parallel loop calls
StatementMatcher galoisLoop = callExpr(callee(functionDecl(anyOf(hasName("for_each"), hasName("do_all"))).bind("gLoopType"))).bind("gLoop");

//Matches calls to getdat and binds getDataVar to the valiable expression which is the node
//g.getData(foo) // binds here, foo bound to getDataVar
StatementMatcher findGetDataAll = memberCallExpr(callee(functionDecl(hasName("getData"))), hasArgument(0, expr().bind("getDataVar")), on(anyOf(memberExpr(hasDeclaration(decl().bind("graphVar"))), declRefExpr(hasDeclaration(decl().bind("graphVar"))))));
//requires the node to be a variable
StatementMatcher findGetDataDirectVar    = memberCallExpr(callee(functionDecl(hasName("getData"))), hasArgument(0, declRefExpr().bind("getDataVar")), on(anyOf(memberExpr(hasDeclaration(decl().bind("graphVar"))), declRefExpr(hasDeclaration(decl().bind("graphVar"))))));

//finds field indexing of references to graph nodes
//N& = g.getData(foo); // foo bound to getDataVar
//N.f; // binds here to fieldRef
StatementMatcher findFieldOfNodeRef = memberExpr(has(declRefExpr(to(varDecl(hasInitializer(findGetDataAll)))))).bind("fieldRef");

//finds references to fields of a graph node
//N& = g.getData(foo).f // foo bound to getDataVar, fieldRef bound here
//N // binds here
StatementMatcher findRefOfFieldRef = declRefExpr(to(varDecl(hasInitializer(memberExpr(has(findGetDataAll)).bind("fieldRef"))))).bind("fieldUse");

//finds direct uses of fields not assigned to references
// g.getData(foo).f 
StatementMatcher findFieldUseDirect = memberExpr(has(findGetDataAll), unless(hasAncestor(varDecl(hasType(referenceType()))))).bind("fieldRef");

//Match all uses of node data
StatementMatcher allFields = anyOf(findFieldOfNodeRef, findRefOfFieldRef, findFieldUseDirect);



//callExpr(callee(functionDecl(anyOf(hasName("for_each"), hasName("do_all")))), hasArgument(2, expr().bind("op")))
//.bind("gLoopType"))).bind("gLoop");
