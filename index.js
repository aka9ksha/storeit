const express = require('express');
const app = express();

app.get('/', (req, res)=>{
    res.send("We are live")
});
app.listen(3333, ()=>console.log("Server is live at port 3333"))

handleNewUserMessage = (newMessage) => {
    fetch('http://localhost:5005/webhooks/rest/webhook', {
          method: 'POST',
           headers : new Headers(),
           body:JSON.stringify({"sender": "USER", "message": newMessage}),
           }).then((res) => res.json())
           .then((data) => {
            var first = data[0];
            var mm= first.text;
            var i;
            console.log(mm)
            toggleMsgLoader();
            setTimeout(() => {
            toggleMsgLoader();
          if (data.length < 1) {
             addResponseMessage("I could not get....");
          } else {
          //if we get response from Rasa
          for ( i = 0; i < data.length; i++) {
              //check if there is text message
              if (data[i].hasOwnProperty("text")) {
                   addResponseMessage(data[i].text );
              }
              if (data[i].hasOwnProperty("buttons")) {
                   setQuickButtons([ { label: data[i].buttons[0].title, value: 'Yes' }, { label: 
                                          data[i].buttons[1].title, value: 'No' } ]);
                             }
              if (data[i].hasOwnProperty("image")) {
                  this.setState({todos: data[i].image})
  
               }
             }
           }
        }, 2000);
     })
     }
  
     handleQuickButtonClicked = (e) => {
         addResponseMessage('Selected ' + e);
         setQuickButtons([]);
     }
   render() 
     return (
       <div>
          <Widget
               title="UMIT CHATBOT"
               handleNewUserMessage={this.handleNewUserMessage}
               handleQuickButtonClicked={this.handleQuickButtonClicked}
               badge={1}
               customComponent={(customData) => (<div><img src={todoss.image} /></div>) }
    />
     </div>
    );
  