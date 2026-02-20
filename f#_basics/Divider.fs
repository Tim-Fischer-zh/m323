module Divider

let divider  f s x =  //ungeföhr das selbe wie public static string divider(int f, int s, int x) 
    match (x % f, x % s) with //stärkeres switch-case, tupel wegen () Beide Modulo füllen dann int (ihre reste) in ihre spalte ein
    |   (0,0) -> "Divided by both" //wenn beides null ist
    |   (_,0) -> "Divided by second"
    |   (0,_) -> "Divided by first"
    |   _ -> string x // alles andere hier raus 