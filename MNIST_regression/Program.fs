open FSharp.Data
open System
let rnd = new Random()

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


let normal (rnd: Random) = 
     let u1 = rnd.NextDouble()

     let u2 = rnd.NextDouble()
     //box muller 
     Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2)

let W1 = Array2D.init 128 784 (fun i j -> normal rnd * Math.Sqrt(2.0 / 784.0))
let b1: float array = Array.zeroCreate 128

let W2 = Array2D.init 10 128 (fun i j -> normal rnd * Math.Sqrt(2.0 / 128.0)) 
let b2: float array = Array.zeroCreate 10

//forward pass braucht skalar produkt