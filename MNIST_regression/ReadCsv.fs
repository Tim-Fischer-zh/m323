module ReadCsv
open FSharp.Data

let read (filename: string) =
    try
        let file = CsvFile.Load(filename)
        printfn "file opened"
        printfn "%A" file.NumberOfColumns
        printfn "%A" file.Rows
        Some file
    with
    | ex ->
        printfn "Exception %s" (ex.Message)
        None