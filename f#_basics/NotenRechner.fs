module NotenRechner



//Note =
// Erzielte Punktezahl × 5 / Max. mögliche Punktezahl + 1
// 




let calculateGrade x =
    match x with
    | x when x > 75 -> 6
    | x when x > 60 -> 5
    | _ -> 1
