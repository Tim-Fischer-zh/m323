

open FSharp.Data
let onehot label = Array.init 10 (fun i -> if i = label then 1.0 else 0.0) 

let readAndNormalizeData (filename: string) =
    //file einlesen
    let file = CsvFile.Load(filename)
    //alle rows
    file.Rows
    //einzelne row die labels und pixels(values) trennen
    |> Seq.map (fun row -> 
        let label =int row.Columns.[0]
        //pixel normalisieren 
        let pixels = row.Columns[1..]|> Array.map (fun p -> float p / 255.0)
        onehot label, pixels)
    |> Seq.toArray
    
    
let data = readAndNormalizeData "/Users/tim/Documents/GitHub/m323/MNIST_regression/data/mnist_test.csv"  


printfn "%A" data[0..5]
