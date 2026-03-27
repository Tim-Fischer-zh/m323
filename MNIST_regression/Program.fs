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
let data: (float array * float array) array = readAndNormalizeData "/Users/tim/Documents/repos/m323/MNIST_regression/data/mnist_train.csv"  

let labels = fst data.[0]
let pixels = snd data.[0]

let normalDistribution (rnd: Random) = 
     let u1 = rnd.NextDouble()

     let u2 = rnd.NextDouble()
     //box muller 
     Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2)

let W1 = Array2D.init 128 784 (fun i j -> normalDistribution rnd * Math.Sqrt(2.0 / 784.0))
let b1: float array = Array.zeroCreate 128

let W2 = Array2D.init 10 128 (fun i j -> normalDistribution rnd * Math.Sqrt(2.0 / 128.0)) 
let b2: float array = Array.zeroCreate 10

let matVecMul (W: float[,]) (x: float array) : float array =
    let rows = Array2D.length1 W
    let cols = Array2D.length2 W
    Array.init rows (fun i ->
        Array.init cols (fun j -> W.[i, j] * x.[j])
        |> Array.sum
    )

let vecAdd (a: float array) (b: float array) : float array= 
    Array.map2 (fun x y  -> x + y) a b
let vecSub (a: float array) (b: float array) : float array= 
    Array.map2 (fun x y  -> x - y) a b
let matTransVecMul (W: float[,]) (x: float array) : float array =
    let cols = Array2D.length2 W 
    let rows = Array2D.length1 W
    Array.init cols (fun i -> 
        Array.init rows (fun j -> W.[j,i] * x.[j])
        |> Array.sum
    )
let vecMul (a: float array) (b: float array) : float array= 
    Array.map2 (fun x y  -> x * y) a b
let relu (a: float array) = 
    Array.map (fun i -> if i < 0.0 then 0.0 else i) a

let reluDerivative (a: float array) = 
    Array.map (fun i -> if i < 0.0 then 0.0 else 1.0) a
let softmax (z: float array) = 
    let maxZ = Array.max z
    let expZ = Array.map (fun j -> exp(j - maxZ)) z 
    let sumExp = Array.sum expZ
    let result = Array.map (fun i -> i / sumExp) expZ 
    result
    
let loss (yHat: float array) (label: float array) : float =
        let i = Array.findIndex(fun x -> x = 1.0 ) label
        - Math.Log(yHat[i])

let backpropagation (yHat: float array) (y: float array) =
    vecSub yHat y 

let outerProduct (a: float array) (b: float array) : float[,] =
    Array2D.init (Array.length a) (Array.length b) (fun i j -> a.[i] * b.[j])




let z1 = matVecMul W1 pixels |> vecAdd b1   
let a1 = relu z1
let z2 = matVecMul W2 a1 |> vecAdd b2
let yHat = softmax z2

let delta2 = vecSub yHat labels
let dW2 = outerProduct delta2 a1
let db2 = delta2
let W2T = matTransVecMul W2 delta2
let g = reluDerivative z1
let delta1 = vecMul W2T g 
let dW1 = outerProduct delta1 pixels
let db1 = delta1