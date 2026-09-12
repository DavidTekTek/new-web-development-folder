// ============================
// MONEYMATE JAVASCRIPT
// ============================

let transactions = [];


// ============================
// ADD TRANSACTION
// ============================

function addTransaction() {

    // Get the information entered
    // by the user.

    let description =
        document.getElementById("description").value;

    let amount =
        Number(document.getElementById("amount").value);

    let type =
        document.getElementById("type").value;


    // STUDENT TASK:
    // Check if description is empty
    // OR amount is zero.
    //
    // If either is true, display:
    //
    // "Please enter valid information."
    //
    // Hint:
    //
    // if(description === "" || amount <= 0)


    transactions.push({
        description: description,
        amount: amount,
        type: type
    });

    displayTransactions();

}


// ============================
// DISPLAY TRANSACTIONS
// ============================

function displayTransactions() {

    let list =
        document.getElementById("transactionList");

    list.innerHTML = "";

    for (let i = 0; i < transactions.length; i++) {

        list.innerHTML +=
            "<li>" +
            transactions[i].description +
            " - ₦" +
            transactions[i].amount +
            " - " +
            transactions[i].type +
            "</li>";

    }

}


// ============================
// SAVINGS CALCULATOR
// ============================

function calculateSavings() {

    let goal =
        Number(document.getElementById("goal").value);

    let saved =
        Number(document.getElementById("saved").value);

    if (goal <= 0 || saved < 0) {

        document.getElementById("savingResult").innerHTML =
            "Please enter valid amounts.";

        return;

    }

    let percentage =
        (saved / goal) * 100;

    if (percentage > 100) {

        percentage = 100;

    }

    document.getElementById("progressBar").style.width =
        percentage + "%";

    document.getElementById("savingResult").innerHTML =
        "You have saved " +
        percentage.toFixed(1) +
        "% of your goal.";

}