$('#listElementDocs').click(function(){
    alert("helloworld");
    $('#docs').replaceWith($('#demo'));
});
$('#listElementDemo').click(function(){
    alert("well");
    $('#demo').replaceWith($('#docs'));
});

