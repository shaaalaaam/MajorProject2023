const menuIcon = document.createElement('span');
menuIcon.id = 'menu-icon';
menuIcon.innerHTML = '&#9776;';
document.body.appendChild(menuIcon);

const sideNav = document.getElementById('side-nav');
const homeBtn = document.getElementById('home-btn');
const simulationBtn = document.getElementById('simulation-btn');
const resultsBtn = document.getElementById('results-btn');

menuIcon.addEventListener('click', () => {
  sideNav.classList.toggle('show');
});

homeBtn.addEventListener('click', () => {
  setActiveButton(homeBtn);
  // Code to show home content goes here
});

simulationBtn.addEventListener('click', () => {
  setActiveButton(simulationBtn);
  // Code to show simulation content goes here
});

resultsBtn.addEventListener('click', () => {
  setActiveButton(resultsBtn);
  // Code to show results content goes here
});

function setActiveButton(btn) {
  // Remove the active class from all buttons
  document.querySelectorAll('#side-nav button').forEach(button => {
    button.classList.remove('active');
  });
  
  // Add the active class to the clicked button
  btn.classList.add('active');
}
