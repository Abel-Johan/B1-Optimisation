x = [-2:0.01:2];
y = [-2:0.01:2]';

f = 100*(y-x.^2).^2 + (1-x).^2;

figure
surf(x, y, f, 'edgecolor', 'none')
xlabel('x-axis');
ylabel('y-axis');
zlabel('f(x, y)');
colorbar

hold on
plot3(1, 1, 0, '.r', 'MarkerSize', 20)
legend('Surf Plot of F','Global Minimum (1, 1)', 'north')

title('Surface Plot of the Rosenbrock Function');

figure
imagesc(x, y, f)
set(gca, 'YDir', 'normal')
xlabel('x-axis');
ylabel('y-axis');
zlabel('f(x, y)');
colorbar

hold on
plot(1, 1, '.r', 'MarkerSize', 20)
legend('Global Minimum (1, 1)', 'north')

title('2D Plot of the Rosenbrock Function');