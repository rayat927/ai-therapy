const express = require('express')
const io = require("socket.io")(4000, {
    cors: {
        origin: "http://localhost:3000",
    }
})
const axios = require('axios')

io.on("connection", (socket) => {
    function getBotMsg(msg){
        axios.post('http://127.0.0.1:5000/text', {text: msg}).then(res => {
            return res.data
        }).catch(err => {
            return err
        })
    }
    socket.on('connect', () => {
        console.log(`connected with ${socket.id}`);
    })

    socket.on('send message', (data) => {
        axios.post('http://127.0.0.1:5000/text', {text: data.msg}).then(res => {
            socket.emit('send message', res.data)
            console.log(res.data);
        }).catch(err => {
            console.log(err); 
        })
    })
}) 