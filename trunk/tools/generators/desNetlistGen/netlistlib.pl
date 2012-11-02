my $MAX_INTER = 100;
my $FinishTime = 100000;

sub genPortVector {
   my ($prefix,$numBits) = @_;

   my $str="";
   my $sep = "";

   if( $numBits == 1 ) {
      $str = $prefix;
   }
   else {
      foreach my $i ( 0..$numBits-1 ) {
         $str =   $str.$sep."${prefix}$i";
         $sep=", ";
      }
   }

   return $str;
}

sub genInitList1bit {
   my ($sigName, $val) = @_;
   my $t = int( rand()*$MAX_INTER + 1 ) ; # preventing time to be zero
   return "initlist $sigName 0, 0 $t, $val end\n";

}

# params
#
# numBits
# prefix  string that is prefix of a vector e.g. for a0,a1,a2,...,a7  'a' is the prefix
# val integer value the vector is to be initialized to 
# numEvents  the number of events/changes to the input  before the final event assigning val occurs
# maxInter   max interval between changes to the input 
 
sub genInitListNbit {

   my ( $numBits, $prefix, $val) = @_[0..2]; # first 3 args
   # shift 3 times now 
   shift @_; shift @_; shift @_;

   my $numEvents = ( @_ != 0 ) ? shift @_ : 0 ; # by default
   my $maxInter = (@_) ? shift @_ : $MAX_INTER ; 


   my $binStr = sprintf("%0${numBits}b",$val);

   if( $DEBUG ) { print "$binStr\n"; }

   my @initList;

   foreach my $i( 0..$numBits-1 ) {
      my $bit = substr( $binStr , length($binStr)-$i-1 , 1 );


      my $init = "initlist ${prefix}$i 0,0 ";

      
      my $prevVal = 0;
      my $prevTime = 0; 
      foreach (1..$numEvents) { 
         $prevVal = 1 - $prevVal ; # toggle value b/w 1 and 0
         $prevTime = int( rand()*$MAX_INTER + 1 ) + $prevTime;
         $init = $init." $prevTime,$prevVal \n";
      }
      
      # now add the actual final time,value pair
      my $rt = int( rand()*$MAX_INTER + 1 ) + $prevTime ; # preventing time to be zero
      $init = $init." $rt,$bit end\n";
      push @initList , $init;
   }

   return \@initList;
}


sub genOutValuesNbit {
   my ($numBits,$outPre,$outVal) = @_;

   my $binStr = sprintf("%0${numBits}b",$outVal);

   if( 0 ) { print "$binStr\n"; }

   my @outValList;


   foreach my $i ( 0..$numBits-1 ) {
      my $bit = substr( $binStr , length($binStr)-$i-1 , 1 );
      my $str = $outPre.$i." ".$bit. ", ";
      push @outValList, $str;
   }

   return \@outValList;
}

sub genOutValues1bit {
   my ($outName,$outVal)= @_;
   my $str = "$outName $outVal, ";
   return $str;
}


sub genKoggeStoneTest {
   my ($FH, $numBits,$s,$cout,$a,$b,$cin)=@_;

   print $FH "\nfinish $FinishTime\n";

   my $inputAList = genPortVector("a",$numBits);
   my $inputBList = genPortVector("b",$numBits);
   print $FH "\ninputs cin, $inputAList, $inputBList\nend\n";


   my $outputList = genPortVector("s", $numBits );
   print $FH "\noutputs cout, $outputList \nend\n";

   my $initListA = genInitListNbit($numBits, "a", $a, 256 );
   my $initListB = genInitListNbit($numBits, "b", $b, 256 );
   my $initListCin = genInitList1bit("cin", $cin);


   print $FH "\n@$initListA\n";
   print $FH "\n@$initListB\n";
   print $FH "\n$initListCin\n";


   my $sOutValues = genOutValuesNbit( $numBits, "s", $s );
   my $coutOutValues = genOutValues1bit( "cout", $cout );

   print $FH "\noutvalues\n";
   print $FH "@$sOutValues\n";
   print $FH ", $coutOutValues\n";
   print $FH "end\n";

   genKoggeStoneNetlist($FH, $numBits);
}

sub genMultTest { 
   my ($FH, $numBits, $mVal, $aVal, $bVal ) = @_;

   print $FH "\nfinish $FinishTime\n";

   # declare the inputs
   print $FH "\ninputs\n";
   foreach ( 0..$numBits-1 )  {
      print $FH "a$_, ";
   }
   print $FH "\n";
   foreach ( 0..$numBits-1 ) {
      print $FH "b$_, ";
   }

   # declare the special input GND
   print $FH "\nGND";
   print $FH "\nend\n";

   # declare the outputs
   print $FH "\noutputs\n";
   foreach  ( 0..2*$numBits-1 ) {
      print $FH "m$_, ";
   }
   print $FH "\nend\n";

   # declare the outvalues

   my $outVals = genOutValuesNbit( 2*$numBits, "m", $mVal );
   print $FH "\noutvalues\n";
   print $FH "@$outVals\n";
   print $FH "end\n";

   # declare the initlist
   # for GND
   print $FH "\ninitlist GND 0 0 end\n";

   my $aInitArray = genInitListNbit( $numBits, "a", $aVal );
   my $bInitArray = genInitListNbit( $numBits, "b", $bVal );

   print $FH "@$aInitArray\n";
   print $FH "@$bInitArray\n";

   genTreeMultUnsigned( $FH, $numBits);
}





1; # return value good
