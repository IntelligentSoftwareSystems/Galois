divert(-1)
# forloop(i, from, to, stmt)
define(`forloop', `pushdef(`$1', `$2')_forloop($@)popdef(`$1')')
define(`_forloop',
       `$4`'ifelse($1, `$3',`', `define(`$1', incr($1))$0($@)')')
divert`'dnl

