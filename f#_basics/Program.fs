open System //using in csharp
open Divider


printf "Choose list length: " 
let l: int = Console.ReadLine() |> int // selbe wie var f = Convert.ToInt32(Console.Readline())

printf "Choose the first divider: " 
let f: int = Console.ReadLine() |> int // selbe wie var f = Convert.ToInt32(Console.Readline())

printf "Choose the second divider: "
let s: int = Console.ReadLine() |> int // |> sind pipes sie nehmen den Wert Links (oder oben) und verarbeiten ihn mit den Angaben in der Pipe, hier z.B. konvertierung in int



// generiert Liste 1 -> 20 |> (pipe) List.map transformiert in eine neue Liste, sieht aber für Entwicklung aus wie (for i in range) loop, weil pipe sich auf die Liste bezieht. 
// (fizzbuzz f s) fizzbuzz siehe oben verlangt 3 Argumente aber wir geben nur 2 mit (f und s) List.map schreib seinen iterativen Wert (x) mit ins fizzbuzz f s 
//pipe für print %A ist das Any Type erwartet wird
[1..l] |> List.map (divider f s)|> List.iter (printfn "%s")