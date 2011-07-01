include(`forloop.m4')
include(`adder_1bit.m4')
define(N,1)

finish=2000

inputs cin, forloop(`i',`0',N, ``a'i,  ')
end

inputs forloop(`i',`0',N, ``b'i,  ')
end

outputs `r'N forloop(`i',`0',N, ``s'i,  ')
end

initlist cin 
0,0
21,1
end

forloop(`i',`0', N, `initlist `a'i 
0,1
end
')

forloop(`i',`0', N, `initlist `b'i 
0,0
end
')

netlist
adder_1bit(s0,r0,a0,b0,cin)
end

forloop(`i',`1', N,`
netlist 
adder_1bit(`s'i,`r'i,`a'i,`b'i,`r'eval(i-1))
end
')
