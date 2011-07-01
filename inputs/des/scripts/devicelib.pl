my $Delay=5;

sub genAdder1bit {

   my ($s,$cout,$a,$b,$cin)=@_;

   my $adder1Bit = <<END_HERE;

inv(${a}n,${a})#4
inv(${b}n,${b})#4
inv(${cin}n,${cin})#4

and2(${a}n${b}n,${a}n,${b}n)#2
and2(${a}n${b},${a}n,${b})#2
and2(${a}${b}n,${a},${b}n)#2
and2(${a}${b},${a},${b})#3

and2(${a}${b}${cin},${a}${b},${cin})#2
and2(${a}n${b}n${cin},${a}n${b}n,${cin})#2
and2(${a}n${b}${cin}n,${a}n${b},${cin}n)#2
and2(${a}${b}n${cin}n,${a}${b}n,${cin}n)#2

and2(${b}${cin},${b},${cin})#2
and2(${a}${cin},${cin},${a})#2

or2(${s}w42,${a}${b}n${cin}n,${a}n${b}${cin}n)#3
or2(${s}w71,${a}${b}${cin}, ${a}n${b}n${cin})#3
or2(${s},${s}w42,${s}w71)#3

or2(${s}w35, ${b}${cin}, ${a}${cin})#3
or2(${cout},${s}w35,${a}${b})#5

END_HERE

   return $adder1Bit;
}



sub genAdder1bitInst {
   my ($inst, $s, $cout, $a, $b, $cin ) = @_;

   my $str = <<END_HERE;
// adder1bit $inst( $s, $cout, $a, $b, $cin)

// sum 
nand2( w0.$inst, $a, $b )#$Delay
nand2( w1.$inst, $a, w0.$inst )#$Delay
nand2( w2.$inst, $b, w0.$inst )#$Delay
nand2( w3.$inst, w1.$inst, w2.$inst )#$Delay

nand2( w4.$inst, $cin, w3.$inst )#$Delay
nand2( w5.$inst, $cin, w4.$inst )#$Delay
nand2( w6.$inst, w3.$inst, w4.$inst )#$Delay
nand2( $s, w5.$inst, w6.$inst )#$Delay

//cout
and2( w7.$inst, $a, $b )#$Delay
and2( w8.$inst, $a, $cin )#$Delay
and2( w9.$inst, $cin, $b )#$Delay

or2( w10.$inst, w7.$inst, w8.$inst  )#$Delay
or2( $cout, w9.$inst, w10.$inst  )#$Delay

// end adder1bit
END_HERE


   return $str;
}

sub genAdder1bitSimple {

   my ($inst, $s, $cout, $a, $b, $cin ) = @_;

   my $str = <<END_HERE;
// adder1bit $inst( $s, $cout, $a, $b, $cin)

// sum 
xor2( w0.$inst, $a, $b )#$Delay
xor2( $s, w0.$inst, $cin )#$Delay

//cout
and2( w1.$inst, $a, $b )#$Delay
and2( w2.$inst, w0.$inst, $cin )#$Delay
or2( $cout, w1.$inst, w2.$inst  )#$Delay

// end adder1bit
END_HERE


   return $str;
}


sub genKoggeStoneNetlist {

   my ($FH, $numBits) = @_;

   my $aVec = [ map { "a$_"; } (0..$numBits-1) ];
   my $bVec = [ map { "b$_"; } (0..$numBits-1) ];
   my $sVec = [ map { "s$_"; } (0..$numBits-1) ];

   genKoggeStoneVec( $FH, $numBits, $sVec, "cout", $aVec, $bVec, "cin" );

}



sub genKoggeStoneVec {


   my ($FH, $numBits, $sVec,$cout,$aVec,$bVec,$cin) = @_;


   foreach my $i (0..$numBits-1) {
      print $FH "\nnetlist\n";
      print $FH (PGgen("g$i", "p$i", $aVec->[$i], $bVec->[$i] ));
      print $FH "\nend\n";
   }



   my $M = 1;
   while( $M < $numBits ) {

      #Gcombine
      foreach ( my $j = 0; $j < $M && ($j+$M) < $numBits ; ++$j ) {
         my $i = $j+$M-1;
         my $k = $j;


         my $g_i_j   = "g_${i}_$cin";
         my $g_i_k   = "g_${i}_${k}";
         my $g_k_1_j = "g_".($k-1)."_$cin";
         my $p_i_k   = "p_${i}_${k}";

         if( $j == 0 ) {
            $g_k_1_j = "$cin";
         }
         if( $M == 1 ) {
            $g_i_k = "g0";
            $p_i_k = "p0";
         }

         print $FH "\nnetlist\n";
         print $FH (Gcombine( $g_i_j, $g_i_k, $g_k_1_j, $p_i_k ) );
         print $FH "\nend\n";
      }


      #PGcombine
      for (my $j=0; ($j <= $numBits-2*$M) && (2*$M <= $numBits) ; ++$j) {
         my $i = $j+2*$M-1;
         my $k = $j+$M;

         my $g_i_j = "g_${i}_${j}";
         my $p_i_j = "p_${i}_${j}";
         my $g_i_k = "g_${i}_${k}";
         my $p_i_k = "p_${i}_${k}";
         my $g_k_1_j = "g_".($k-1)."_$j";
         my $p_k_1_j = "p_".($k-1)."_$j";

         if( $M==1 ) {

            $g_i_k = "g$i";
            $p_i_k = "p$i";
            $g_k_1_j = "g$j";
            $p_k_1_j = "p$j";
         }

         print $FH "\nnetlist\n";
         print $FH (PGcombine( $g_i_j, $p_i_j, $g_i_k, $g_k_1_j, $p_i_k, $p_k_1_j) );
         print $FH "\nend\n";
      }


      # update M
      $M = 2*$M;
   }

   





   # when $numBits is not a power of 2, $M exceeds it in the 
   # while loop above
   if( $M > $numBits ) {
      $M = $M/2;
   }

   # cout logic
   my $g_i_k = "g_".($numBits-1)."_".($numBits-$M);
   my $p_i_k = "p_".($numBits-1)."_".($numBits-$M); 
   my $g_k_1_j = "g_".($numBits-$M-1)."_$cin";
   if( $numBits == $M ) {
      $g_k_1_j = "$cin";
   }

   print $FH "\nnetlist\n";
   print $FH (
      Gcombine(
         $cout, 
         $g_i_k,
         $g_k_1_j, 
         $p_i_k, 
      )
   );

   print $FH "\nend\n";

   # sum array
   print $FH "\nnetlist\n";
   foreach my $i( 0..$numBits-1 ) {
      if( $i == 0 ) {
         print $FH "\nxor2( $sVec->[$i], p$i,$cin )#$Delay \n"; 
      }
      else {
         print $FH "\nxor2( $sVec->[$i], p$i,g_".($i-1)."_$cin )#$Delay\n"; 
      }
   }
   print $FH "\nend\n";
   

}


sub PGgen {
   my ($g,$p,$a,$b) = @_;

   my $str = <<HERE;
// PGgen ($g,$p,$a,$b)
   xor2($p,$a,$b)#$Delay
   and2($g,$a,$b)#$Delay

HERE

   return $str;
}


sub Gcombine {
   my ($g_i_j, $g_i_k, $g_k_1_j, $p_i_k ) = @_;

   my $str = <<HERE;
// Gcombine ($g_i_j, $g_i_k, $g_k_1_j, $p_i_k )
   and2(w0$g_i_j, $p_i_k, $g_k_1_j )#$Delay
   or2( $g_i_j, w0$g_i_j, $g_i_k )#$Delay

HERE

   return $str;
}


sub PGcombine {
   my ($g_i_j,$p_i_j,$g_i_k,$g_k_1_j,$p_i_k,$p_k_1_j) = @_;

   my $str = <<HERE;
// PGcombine ($g_i_j,$p_i_j,$g_i_k,$p_i_k,$g_k_1_j,$p_k_1_j)
   and2(  w0$g_i_j, $p_i_k, $g_k_1_j )#$Delay
   or2( $g_i_j,  w0$g_i_j, $g_i_k )#$Delay
   and2( $p_i_j,  $p_i_k, $p_k_1_j)#$Delay

HERE

   return $str;
}


sub xor2x1 {
   my ($inst,$z,$a,$b) = @_;

   my $str = <<END_HERE;
// xor2x1 $inst( $z, $a, $b )

inv( ${a}_n.$inst, $a )#$Delay
inv( ${b}_n.$inst, $b )#$Delay
nand2( w0.$inst, ${a}_n.$inst, $b )#$Delay
nand2( w1.$inst, ${b}_n.$inst, $a )#$Delay
nand2( $z, w0.$inst, w1.$inst );

// end xor2x1

END_HERE

   return $str;
}

sub genTreeMultUnsigned {
   my ($FH, $numBits) = @_;

   my $aVec = [ map { "a$_"; } (0..$numBits-1) ];
   my $bVec = [ map { "b$_"; } (0..$numBits-1) ]; 

   my $zeroVec = [ map { 'GND'; } (0..2*$numBits-1) ]; # twice the numBits to be used in muxes

   my $shiftedVecList = [];


   # create the shifted-concatenated aVec list
   foreach my $j ( 0..$numBits-1 ) {

      push @$shiftedVecList, concatShift( $numBits, $aVec, $j );

   }

   # mux the shiftedVecList to generate partial products

   my @partProdList = ();
   foreach my $i  ( 0..$#$shiftedVecList ) {

      my $shiftedVec = $shiftedVecList->[$i];
      
      my $zVec = [ map { "PP_${i}_$_"; } (0..2*$numBits-1) ];

      my $muxInstVec = mux2x1Vec( "Bth$i", $zVec, $zeroVec, $shiftedVec, $bVec->[$i] );

      foreach my $m ( @$muxInstVec ) {
         print $FH "\nnetlist\n";
         print $FH "$m";
         print $FH "\nend\n";
      }

      push @partProdList, $zVec;
   }


   my $outVec = [ map { "m$_"; } (0..2*$numBits-1) ];

   
   csaTree4x2( $FH, $numBits, \@partProdList, $outVec );

   
}

sub mux2x1 {
   my ($inst, $z,$i0,$i1,$s) = @_;

my $str = <<END_HERE;

// mux2x1 $inst( $z, $i0, $i1, $s )

inv(${s}_n.$inst, $s )#$Delay
and2(w0.$inst, ${s}_n.$inst, $i0 )#$Delay
and2(w1.$inst, $s, $i1 )#$Delay
or2($z, w0.$inst, w1.$inst )#$Delay

// end mux2x1 $inst

END_HERE

return $str;
}


sub mux2x1Nbit {
   my ($numBits, $inst, $z, $i0, $i1, $s ) = @_;

   my @array=();

   foreach my $i( 0..$numBits-1 ) {
      my $str = mux2x1( "$inst.$i", $z.$i, $i0.$i, $i1.$i, $s );
      push @array, $str;
   }

   return \@array;
}

sub mux2x1Vec {
   my ( $inst, $zVec, $i0Vec, $i1Vec, $s ) = @_;

   my @strVec = ();
   foreach my $i ( 0..$#$zVec ) {
      my $str = mux2x1( "$inst.$i", $zVec->[$i], $i0Vec->[$i], $i1Vec->[$i] , $s );
      push @strVec, $str;
   }

   return \@strVec;
}


sub concatShift {
   my ($numBits, $inVec, $shift ) = @_;

   my $outVec = [ @$inVec ];

   foreach ( 1..$shift ) {
      unshift @$outVec, 'GND';
   }

   foreach ( 1..$numBits-$shift ) {
      push @$outVec, 'GND';
   }

   return $outVec;
}


sub csa4x2 {
   my ( $inst, $s,$t,$cout,$a,$b,$c,$d,$cin) = @_;

   my $str1 = genAdder1bitSimple( "Adder0.$inst", "sint".$inst, $cout, $a,$b,$c );
   my $str2 = genAdder1bitSimple( "Adder1.$inst", $s, $t, "sint".$inst, $d, $cin ); 


   my $str = <<END_HERE;
   // csa4x2 ($inst, $s,$t,$cout,$a,$b,$c,$d,$cin) ;


   $str1

   $str2

   // end csa4x2

END_HERE


   return $str;
}


sub csa4x2Vec {

   my ( $instPre, $sVec, $tVec, $aVec, $bVec, $cVec, $dVec ) = @_;

   my @outputVec = ();


   for my $i ( 0..$#$aVec ) {

      my $cin;
      my $coutPre =  "cout_$instPre.";

      if( $i == 0 ) {
         $cin = 'GND';
      }
      else {
         $cin = $coutPre.($i-1);
      }
      
      my $str = csa4x2( 
         $instPre.".".$i,

         $sVec->[$i], 
         $tVec->[$i], 
         $coutPre.$i, 
         $aVec->[$i],
         $bVec->[$i],
         $cVec->[$i],
         $dVec->[$i],
         $cin,
      );


      $outputVec[$i] = $str;
   }


   return \@outputVec;
}


# 
# generate a csa4x2 tree for partial products for example
#
# inputs
# 
# numBits
# list of partial product vectors
# a vector of output names
# 
# 
#



sub csaTree4x2 {

   my ($FH, $numBits, $partProdList, $outVec) = @_;

   my $j = $numBits; # number of partial products

   my $currRow = $partProdList;

   my $lvl = 0;

   while( $#$currRow+1 >= 4  ) {  # keep instantiating rows of csa4x2
      
      my $newRow = [];


      print $FH "\n// instantiating csa4x2Vec at lvl $lvl \n";


      for ( my $i = 0; $#$currRow+1 >= 4 ; $i+=4 ) { 

         my $tVec = [ map { "t${i}_lvl".$lvl."_$_"; } (0..2*$numBits-1) ];
         my $sVec = [ map { "s${i}_lvl".$lvl."_$_"; } (0..2*$numBits-1) ];


         my $instList = csa4x2Vec( "csa4x2_${i}_lvl${lvl}", $sVec, $tVec, $currRow->[0], $currRow->[1], $currRow->[2], $currRow->[3] );


         foreach my $inst ( @$instList ) {
            print $FH "\nnetlist\n";
            print $FH "$inst";
            print $FH "\nend\n";
         }


         # remove 4 elements from currRow
         @$currRow = @$currRow[4..$#$currRow];

         # shift the tVec by 1 to the left
         for( my $k = $#$tVec; $k > 0; --$k ) {
            $tVec->[$k] = $tVec->[$k-1];
         }
         $tVec->[0] = 'GND';

         push @$newRow , $sVec; 
         push @$newRow , $tVec;


      }

      # add elements from newRow to currRow
      push @$currRow, @$newRow;

      $j = $j/2;
      ++$lvl;
   }



   # currRow contains the final sVec and shifted tVec
   # feed into a tree adder.



   print $FH qq{\n// genKoggeStoneVec(2*numBits, outVec, "cout", currRow->[0], currRow->[1], 'GND' );\n};

   genKoggeStoneVec( $FH, 2*$numBits, $outVec, "cout", $currRow->[0], $currRow->[1], 'GND' ); # cin as 0


}

1; # return a true value
