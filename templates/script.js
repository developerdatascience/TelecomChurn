function total(tenure, mcharges){
    var tenure = parseInt(document.getElementById("tenure").value);
    var mcharges = parseInt(document.getElementById("monthlycharges").value);
    var res= tenure*mcharges;
    document.getElementById("totalcharges").value = res;
}
