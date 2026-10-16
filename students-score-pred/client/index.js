console.log('ASS IS LIFE, BUT BOOBS AND THIGHS ARE ENTERNAL')
const form = document.getElementById('form')
const message = document.getElementById('predicted')

// stop form from reloading

form.addEventListener('submit', async (e) =>{
    e.preventDefault()

    console.log('Form did not reload')


    const age = Number(document.getElementById('age').value)
    const absence = Number(document.getElementById('absence').value)
    const fpg = Number(document.getElementById('firstGrade').value)
    const spg = Number(document.getElementById('secondGrade').value)
    const time = Number(document.getElementById('time').value)

    const data = {
        age: age,
        absences: absence,
        G1: fpg,
        G2: spg,
        studytime: time
    }


    const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST', 
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })

    const prediction = await response.json()

    const predict = prediction.prediction

    message.textContent = `Predicted Score: ${predict}`

    // form.reset()
})
