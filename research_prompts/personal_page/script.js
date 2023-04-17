// scripts.js

document.addEventListener("DOMContentLoaded", function() {
    const sectionTitles = document.querySelectorAll("section h2");

    sectionTitles.forEach(title => {
        title.addEventListener("click", function() {
            const content = this.nextElementSibling;
            content.classList.toggle("hidden");
        });
    });
});
