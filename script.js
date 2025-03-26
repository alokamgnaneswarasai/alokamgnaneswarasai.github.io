document.querySelectorAll('nav a').forEach(link => {
    link.addEventListener('click', function (e) {
        e.preventDefault();

        // Get the section ID from the data-section attribute
        const sectionId = this.getAttribute('data-section');

        // Hide all sections except About Me
        document.querySelectorAll('main section').forEach(section => {
            if (section.id !== 'about') {
                section.classList.add('hidden');
            }
        });

        // Show the clicked section
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.classList.remove('hidden');
        }
    });
});